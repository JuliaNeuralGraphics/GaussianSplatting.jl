"""
Camera frustum world pose.

- `model`: maps the frustum's local frame to world space (c2w).
    Columns are the camera basis (`x` right, `y` down, `z` forward)
    and the camera center (COLMAP format).
- `tan_half`: half-extent of the image plane at unit depth, along `x` & `y`.
"""
struct FrustumPose
    model::SMatrix{4, 4, Float32, 16}
    tan_half::SVector{2, Float32}
end

FrustumPose(c::Camera) = FrustumPose(
    SMatrix{3, 3, Float32, 9}(@view(c.c2w[1:3, 1:3])),
    SVector{3, Float32}(@view(c.c2w[1:3, 4])), camera_tan_half(c))

# tan(fov / 2) == (resolution / 2) / focal.
camera_tan_half(c::Camera) = SVector{2, Float32}(
    0.5f0 .* c.intrinsics.resolution ./ c.intrinsics.focal)

# `R` & `t` are the camera's c2w rotation & center, which is what a
# `NU.CameraKeyframe` stores - a path keyframe is drawn without a `Camera`.
function FrustumPose(
    R::SMatrix{3, 3, Float32, 9}, t::SVector{3, Float32},
    tan_half::SVector{2, Float32},
)
    model = SMatrix{4, 4, Float32, 16}(
        R[1, 1], R[2, 1], R[3, 1], 0f0,
        R[1, 2], R[2, 2], R[3, 2], 0f0,
        R[1, 3], R[2, 3], R[3, 3], 0f0,
        t[1], t[2], t[3], 1f0)
    FrustumPose(model, tan_half)
end

# World-space size of the frustum drawn at `scale` depth.
function extent(p::FrustumPose, scale::Float32)
    SVector{3, Float32}(scale * p.tan_half[1], scale * p.tan_half[2], scale)
end

# Offset image plane to avoid z-fighting.
const IMAGE_PLANE_INSET = 0.995f0

# Core profiles are only required to support a line width of `1` and macOS
# raises `GL_INVALID_VALUE` for anything wider instead of clamping to the
# supported range, so ask for the range and stay within it.
const MAX_LINE_WIDTH = Ref{Float32}(0f0)

function max_line_width()
    if MAX_LINE_WIDTH[] == 0f0
        range = Float32[1f0, 1f0]
        glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, range)
        glGetError() # Discard the error if the query is unsupported.
        MAX_LINE_WIDTH[] = max(1f0, range[2])
    end
    MAX_LINE_WIDTH[]
end

"""
Geometry & shaders shared by every camera frustum drawn in the scene:
a wireframe pyramid (with an "up" marker over the top edge) and a textured quad for the view's image.

Both are unit-sized in the frustum's local frame,
`FrustumPose.model` & `extent` place and size them.
"""
struct FrustumRenderer
    line_program::NGL.ShaderProgram
    image_program::NGL.ShaderProgram
    line_va::NGL.VertexArray
    quad_va::NGL.VertexArray
end

struct FrustumQuadVertex
    position::SVector{3, Float32}
    uv::SVector{2, Float32}
end

function FrustumRenderer()
    vertices = SVector{3, Float32}[
        SVector{3, Float32}(0f0, 0f0, 0f0),    # 0: apex
        SVector{3, Float32}(-1f0, -1f0, 1f0),  # 1: top-left
        SVector{3, Float32}( 1f0, -1f0, 1f0),  # 2: top-right
        SVector{3, Float32}( 1f0,  1f0, 1f0),  # 3: bottom-right
        SVector{3, Float32}(-1f0,  1f0, 1f0),  # 4: bottom-left
        # Up-marker: a triangle standing on the top edge.
        SVector{3, Float32}(-0.5f0, -1f0, 1f0),   # 5
        SVector{3, Float32}( 0.5f0, -1f0, 1f0),   # 6
        SVector{3, Float32}( 0f0, -1.5f0, 1f0)]   # 7
    indices = UInt32[
        0, 1, 0, 2, 0, 3, 0, 4, # Apex to the image plane corners.
        1, 2, 2, 3, 3, 4, 4, 1, # Image plane outline.
        5, 7, 6, 7]             # Up-marker sides (its base is the top edge).
    line_va = NGL.VertexArray(
        NGL.IndexBuffer(indices; primitive_type=GL_LINES),
        NGL.VertexBuffer(vertices, NGL.BufferLayout([
            NGL.BufferElement(SVector{3, Float32}, "position")])))

    # `v = 0` is the first uploaded image row, i.e. its top row, which is
    # at `y = -1`: `y` points down in the camera basis.
    quad_vertices = FrustumQuadVertex[
        FrustumQuadVertex(SVector{3, Float32}(-1f0, -1f0, 1f0), SVector{2, Float32}(0f0, 0f0)),
        FrustumQuadVertex(SVector{3, Float32}( 1f0, -1f0, 1f0), SVector{2, Float32}(1f0, 0f0)),
        FrustumQuadVertex(SVector{3, Float32}( 1f0,  1f0, 1f0), SVector{2, Float32}(1f0, 1f0)),
        FrustumQuadVertex(SVector{3, Float32}(-1f0,  1f0, 1f0), SVector{2, Float32}(0f0, 1f0))]
    quad_va = NGL.VertexArray(
        NGL.IndexBuffer(UInt32[0, 1, 2, 2, 3, 0]),
        NGL.VertexBuffer(quad_vertices, NGL.BufferLayout([
            NGL.BufferElement(SVector{3, Float32}, "position"),
            NGL.BufferElement(SVector{2, Float32}, "uv")])))

    line_program = NGL.ShaderProgram((
        NGL.Shader(GL_VERTEX_SHADER, """
        #version 330 core
        layout (location = 0) in vec3 position;

        uniform mat4 proj;
        uniform mat4 view;
        uniform mat4 model;
        uniform vec3 extent;

        void main() {
            gl_Position = proj * view * model * vec4(position * extent, 1.0);
        }
        """),
        NGL.Shader(GL_FRAGMENT_SHADER, """
        #version 330 core

        uniform vec4 u_color;

        layout (location = 0) out vec4 color;

        void main() {
            color = u_color;
        }
        """)))

    image_program = NGL.ShaderProgram((
        NGL.Shader(GL_VERTEX_SHADER, """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 uv;

        uniform mat4 proj;
        uniform mat4 view;
        uniform mat4 model;
        uniform vec3 extent;
        uniform vec3 eye;

        out vec2 v_uv;
        out float v_facing;

        void main() {
            vec4 world = model * vec4(position * extent, 1.0);
            v_uv = uv;
            // Positive when the viewer is past the image plane, looking at
            // its back: the frustum's forward axis is the model's 3rd column.
            v_facing = dot(eye - world.xyz, vec3(model[2]));
            gl_Position = proj * view * world;
        }
        """),
        NGL.Shader(GL_FRAGMENT_SHADER, """
        #version 330 core
        in vec2 v_uv;
        in float v_facing;

        uniform sampler2D u_texture;
        uniform float u_opacity;

        layout (location = 0) out vec4 color;

        void main() {
            vec3 rgb = texture(u_texture, v_uv).rgb;
            // Dim the back side, so which way a camera looks stays readable.
            if (v_facing > 0.0) rgb *= 0.45;
            color = vec4(rgb, u_opacity);
        }
        """)))

    NGL.bind(image_program)
    NGL.upload_uniform(image_program, "u_texture", 0)
    NGL.unbind(image_program)

    FrustumRenderer(line_program, image_program, line_va, quad_va)
end

"""
    draw_wireframes(r, poses, P, V; scale, color, highlight, highlight_color)

Draw the wireframe of every pose in `poses`.

- `P`, `V`: perspective & look-at matrices of the viewing camera.
- `scale`: depth of the drawn frustums, in world units.
- `highlight`: index into `poses` drawn in `highlight_color` (`0` for none).
"""
function draw_wireframes(
    r::FrustumRenderer, poses::AbstractVector{FrustumPose}, P, V;
    scale::Float32, color::SVector{4, Float32},
    highlight::Int = 0,
    highlight_color::SVector{4, Float32} = SVector{4, Float32}(1f0, 0.78f0, 0.25f0, 1f0),
)
    isempty(poses) && return

    p = r.line_program
    NGL.bind(p)
    NGL.bind(r.line_va)
    NGL.upload_uniform(p, "proj", P)
    NGL.upload_uniform(p, "view", V)

    glLineWidth(min(1.5f0, max_line_width()))
    for (i, pose) in enumerate(poses)
        NGL.upload_uniform(p, "model", pose.model)
        NGL.upload_uniform(p, "extent", extent(pose, scale))
        NGL.upload_uniform(p, "u_color", i == highlight ? highlight_color : color)
        NGL.draw(r.line_va)
    end
    glLineWidth(1f0)
    return
end

"""
    draw_images(r, poses, textures, P, V; scale, eye, opacity)

Draw `textures[i]` on the image plane of `poses[i]`. `textures` may be
shorter than `poses` (thumbnails are uploaded incrementally): the frustums
without one are skipped.

`eye` is the world-space position of the viewing camera; it decides which
side of an image plane is being looked at.
"""
function draw_images(
    r::FrustumRenderer, poses::AbstractVector{FrustumPose},
    textures::AbstractVector{NGL.Texture}, P, V;
    scale::Float32, eye::SVector{3, Float32}, opacity::Float32 = 1f0,
)
    isempty(textures) && return

    p = r.image_program
    NGL.bind(p)
    NGL.bind(r.quad_va)
    NGL.upload_uniform(p, "proj", P)
    NGL.upload_uniform(p, "view", V)
    NGL.upload_uniform(p, "eye", eye)
    NGL.upload_uniform(p, "u_opacity", opacity)

    blend = opacity < 1f0
    blend && NGL.enable_blend()
    for i in 1:length(textures)
        pose = poses[i]
        NGL.upload_uniform(p, "model", pose.model)
        NGL.upload_uniform(p, "extent", extent(pose, scale * IMAGE_PLANE_INSET))
        NGL.bind(textures[i], 0)
        NGL.draw(r.quad_va)
    end
    blend && NGL.disable_blend()
    return
end

"""
Frustums of a dataset's training views, with each view's image on the frustum's image plane.

The images are `dataset.train_thumbnails`, downscaled at dataset load time
(see `with_thumbnails`), so building this only costs the texture uploads.
`thumbnails` stays empty for a dataset loaded without them, which just
leaves the frustums without images.
"""
struct CameraFrustums
    poses::Vector{FrustumPose}
    thumbnails::Vector{NGL.Texture}
end

function CameraFrustums(dataset::ColmapDataset)
    poses = [FrustumPose(c) for c in dataset.train_cameras]
    thumbnails = map(dataset.train_thumbnails) do data
        _, width, height = size(data)
        texture = NGL.Texture(width, height;
            internal_format=GL_RGB8, data_format=GL_RGB, type=GL_UNSIGNED_BYTE,
            wrap_s=GL_CLAMP_TO_EDGE, wrap_t=GL_CLAMP_TO_EDGE)
        NGL.set_data!(texture, data)
        texture
    end
    CameraFrustums(poses, thumbnails)
end

Base.length(f::CameraFrustums) = length(f.poses)

function free!(f::CameraFrustums)
    for texture in f.thumbnails
        NGL.delete!(texture)
    end
    empty!(f.thumbnails)
    return
end
