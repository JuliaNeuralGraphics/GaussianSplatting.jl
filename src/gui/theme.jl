# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

# `NGL.Context` starts from ImGui's built-in dark theme, which tints every
# interactive widget blue. `apply_theme!` overwrites the palette with a
# neutral grayscale one, so the only colored things left in the UI are the
# ones where color carries meaning (the accent below, destructive buttons,
# loss curves).

gray(v::Real, a::Real = 1) = CImGui.ImVec4(Float32(v), Float32(v), Float32(v), Float32(a))

# Accent (#ebb434): the only hue in the UI, on the widgets that either
# start something or carry a value.
const ACCENT = CImGui.ImVec4(0.922f0, 0.706f0, 0.204f0, 1f0)
const ACCENT_HOVERED = CImGui.ImVec4(1f0, 0.784f0, 0.278f0, 1f0)
const ACCENT_ACTIVE = CImGui.ImVec4(0.792f0, 0.588f0, 0.129f0, 1f0)

function set_style_color!(colors::Ptr{CImGui.ImVec4}, idx, color::CImGui.ImVec4)
    unsafe_store!(colors, color, Int(idx) + 1)
    return
end

function apply_theme!()
    style = CImGui.GetStyle()
    colors = Ptr{CImGui.ImVec4}(style.Colors)

    set!(idx, color) = set_style_color!(colors, idx, color)

    set!(CImGui.ImGuiCol_Text,                  gray(0.92))
    set!(CImGui.ImGuiCol_TextDisabled,          gray(0.45))
    set!(CImGui.ImGuiCol_WindowBg,              gray(0.07))
    set!(CImGui.ImGuiCol_ChildBg,               gray(0.00, 0.00))
    set!(CImGui.ImGuiCol_PopupBg,               gray(0.09, 0.96))
    set!(CImGui.ImGuiCol_Border,                gray(0.30))
    set!(CImGui.ImGuiCol_BorderShadow,          gray(0.00, 0.00))

    set!(CImGui.ImGuiCol_FrameBg,               gray(0.17))
    set!(CImGui.ImGuiCol_FrameBgHovered,        gray(0.27))
    set!(CImGui.ImGuiCol_FrameBgActive,         gray(0.35))

    set!(CImGui.ImGuiCol_TitleBg,               gray(0.08))
    set!(CImGui.ImGuiCol_TitleBgActive,         gray(0.18))
    set!(CImGui.ImGuiCol_TitleBgCollapsed,      gray(0.05, 0.75))
    set!(CImGui.ImGuiCol_MenuBarBg,             gray(0.11))

    set!(CImGui.ImGuiCol_ScrollbarBg,           gray(0.04, 0.60))
    set!(CImGui.ImGuiCol_ScrollbarGrab,         gray(0.28))
    set!(CImGui.ImGuiCol_ScrollbarGrabHovered,  gray(0.38))
    set!(CImGui.ImGuiCol_ScrollbarGrabActive,   gray(0.48))

    set!(CImGui.ImGuiCol_CheckMark,             gray(0.92))
    set!(CImGui.ImGuiCol_SliderGrab,            ACCENT)
    set!(CImGui.ImGuiCol_SliderGrabActive,      ACCENT_HOVERED)

    set!(CImGui.ImGuiCol_Button,                gray(0.21))
    set!(CImGui.ImGuiCol_ButtonHovered,         gray(0.33))
    set!(CImGui.ImGuiCol_ButtonActive,          gray(0.45))

    set!(CImGui.ImGuiCol_Header,                gray(0.22))
    set!(CImGui.ImGuiCol_HeaderHovered,         gray(0.32))
    set!(CImGui.ImGuiCol_HeaderActive,          gray(0.42))

    set!(CImGui.ImGuiCol_Separator,             gray(0.28))
    set!(CImGui.ImGuiCol_SeparatorHovered,      gray(0.45))
    set!(CImGui.ImGuiCol_SeparatorActive,       gray(0.60))

    set!(CImGui.ImGuiCol_ResizeGrip,            gray(0.25, 0.60))
    set!(CImGui.ImGuiCol_ResizeGripHovered,     gray(0.40, 0.80))
    set!(CImGui.ImGuiCol_ResizeGripActive,      gray(0.55))

    set!(CImGui.ImGuiCol_InputTextCursor,       gray(0.92))

    set!(CImGui.ImGuiCol_Tab,                   gray(0.14))
    set!(CImGui.ImGuiCol_TabHovered,            gray(0.34))
    set!(CImGui.ImGuiCol_TabSelected,           gray(0.26))
    set!(CImGui.ImGuiCol_TabSelectedOverline,   gray(0.75))
    set!(CImGui.ImGuiCol_TabDimmed,             gray(0.09))
    set!(CImGui.ImGuiCol_TabDimmedSelected,     gray(0.18))
    set!(CImGui.ImGuiCol_TabDimmedSelectedOverline, gray(0.35))

    set!(CImGui.ImGuiCol_DockingPreview,        gray(0.60, 0.40))
    set!(CImGui.ImGuiCol_DockingEmptyBg,        gray(0.07))

    set!(CImGui.ImGuiCol_PlotLines,             gray(0.70))
    set!(CImGui.ImGuiCol_PlotLinesHovered,      gray(0.95))
    set!(CImGui.ImGuiCol_PlotHistogram,         gray(0.70))
    set!(CImGui.ImGuiCol_PlotHistogramHovered,  gray(0.95))

    set!(CImGui.ImGuiCol_TableHeaderBg,         gray(0.16))
    set!(CImGui.ImGuiCol_TableBorderStrong,     gray(0.32))
    set!(CImGui.ImGuiCol_TableBorderLight,      gray(0.24))
    set!(CImGui.ImGuiCol_TableRowBg,            gray(0.00, 0.00))
    set!(CImGui.ImGuiCol_TableRowBgAlt,         gray(1.00, 0.04))

    set!(CImGui.ImGuiCol_TextLink,              gray(0.85))
    set!(CImGui.ImGuiCol_TextSelectedBg,        gray(1.00, 0.25))
    set!(CImGui.ImGuiCol_TreeLines,             gray(0.30))

    set!(CImGui.ImGuiCol_DragDropTarget,        gray(1.00, 0.90))
    set!(CImGui.ImGuiCol_NavCursor,             gray(0.70))
    set!(CImGui.ImGuiCol_NavWindowingHighlight, gray(1.00, 0.70))
    set!(CImGui.ImGuiCol_NavWindowingDimBg,     gray(0.20, 0.20))
    set!(CImGui.ImGuiCol_ModalWindowDimBg,      gray(0.10, 0.55))
    return
end

function accent_button_begin()
    CImGui.PushStyleColor(CImGui.ImGuiCol_Button, ACCENT)
    CImGui.PushStyleColor(CImGui.ImGuiCol_ButtonHovered, ACCENT_HOVERED)
    CImGui.PushStyleColor(CImGui.ImGuiCol_ButtonActive, ACCENT_ACTIVE)
    # The accent is bright enough that the default light label washes out.
    CImGui.PushStyleColor(CImGui.ImGuiCol_Text, gray(0.08))
end

function accent_button_end()
    CImGui.PopStyleColor(4)
end

"""
Segmented control: one button per entry of `names`, laid out side by side,
with the active one carrying the accent. `mode` holds the active index,
0-based to match the `CImGui.Combo` these replace.

Returns `true` on the frame the selection changes.
"""
function mode_buttons!(id::String, mode::Ref{Int32}, names::Vector{String})
    changed = false
    CImGui.BeginTable(id, length(names))
    CImGui.TableNextRow()
    for (i, name) in enumerate(names)
        value = Int32(i - 1)
        is_active = mode[] == value
        CImGui.TableNextColumn()
        is_active && accent_button_begin()
        if CImGui.Button(name, CImGui.ImVec2(-1, 0)) && !is_active
            mode[] = value
            changed = true
        end
        is_active && accent_button_end()
    end
    CImGui.EndTable()
    return changed
end
