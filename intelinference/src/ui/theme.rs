//! Colours. Truecolor; the screen paints its own background so effects have real colours to blend.

use ratatui::style::{Color, Modifier, Style};

pub const BG: Color = Color::Rgb(9, 13, 24);
pub const PANEL: Color = Color::Rgb(14, 20, 36);
pub const LINE: Color = Color::Rgb(38, 50, 78);
pub const TEXT: Color = Color::Rgb(208, 216, 232);
pub const DIM: Color = Color::Rgb(112, 124, 150);
pub const FAINT: Color = Color::Rgb(64, 76, 102);
pub const BLUE: Color = Color::Rgb(0, 113, 197); // Intel blue
pub const CYAN: Color = Color::Rgb(0, 199, 253);
pub const ICE: Color = Color::Rgb(170, 235, 255);
pub const GREEN: Color = Color::Rgb(84, 222, 145);
pub const AMBER: Color = Color::Rgb(255, 186, 72);
pub const RED: Color = Color::Rgb(255, 96, 96);
pub const VIOLET: Color = Color::Rgb(150, 120, 255);

pub fn text() -> Style {
    Style::new().fg(TEXT).bg(BG)
}
pub fn dim() -> Style {
    Style::new().fg(DIM)
}
pub fn faint() -> Style {
    Style::new().fg(FAINT)
}
pub fn bold(c: Color) -> Style {
    Style::new().fg(c).add_modifier(Modifier::BOLD)
}

/// The colour of a server state word.
pub fn state_color(state: &str) -> Color {
    match state {
        "generating" => CYAN,
        "reading" => VIOLET,
        "idle" => GREEN,
        "unloaded" => AMBER,
        "offline" => RED,
        _ => DIM,
    }
}
