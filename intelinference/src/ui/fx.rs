//! Every animation on the screen, in one place (tachyonfx). Effects run over what was drawn this frame.

use std::time::Duration;

use ratatui::{buffer::Buffer, layout::Rect, style::Color};
use tachyonfx::{
    CellFilter, EffectManager, Interpolation as I, Motion,
    fx::{self, EvolveSymbolSet},
    pattern::{DiagonalDirection, DiagonalPattern, RadialPattern, SweepPattern},
};

use super::{ToastKind, theme};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    #[default]
    Screen,
    Body,
    Badge,
    Speed,
    Header,
    Toast,
    Popup,
}

pub struct Fx {
    m: EffectManager<Key>,
}

fn ok(r: Rect) -> bool {
    r.width > 0 && r.height > 0
}

impl Fx {
    pub fn new() -> Self {
        Self { m: EffectManager::default() }
    }

    pub fn running(&self) -> bool {
        self.m.is_running()
    }

    pub fn process(&mut self, dt: Duration, buf: &mut Buffer, area: Rect) {
        self.m.process_effects(dt.into(), buf, area);
    }

    /// Start-up: the screen gathers out of noise from the centre, the header sweeps in.
    pub fn intro(&mut self, screen: Rect) {
        let gather = fx::coalesce((900, I::ExpoOut)).with_pattern(RadialPattern::center().with_transition_width(6.0));
        let tint = fx::fade_from_fg(theme::CYAN, (1400, I::SineOut));
        self.m.add_unique_effect(Key::Screen, fx::parallel(&[gather, tint]).with_area(screen));
    }

    /// Changing tab: the new view sweeps in from the side it comes from.
    pub fn tab(&mut self, body: Rect, forward: bool) {
        if !ok(body) {
            return;
        }
        let dir = if forward { Motion::LeftToRight } else { Motion::RightToLeft };
        let e = fx::sweep_in(dir, 24, 6, theme::BG, (380, I::QuadOut)).with_area(body);
        self.m.add_unique_effect(Key::Body, e);
    }

    /// The server answered (again): content materialises diagonally.
    pub fn online(&mut self, body: Rect) {
        if !ok(body) {
            return;
        }
        let e = fx::parallel(&[
            fx::coalesce((700, I::QuadOut))
                .with_pattern(DiagonalPattern::new(DiagonalDirection::TopLeftToBottomRight, 8.0)),
            fx::fade_from_fg(theme::ICE, (900, I::SineOut)),
        ])
        .with_area(body);
        self.m.add_unique_effect(Key::Body, e);
    }

    /// The server went away: the view drains to grey (it stays grey: the offline view is drawn in grey).
    pub fn offline(&mut self, body: Rect) {
        if !ok(body) {
            return;
        }
        let e = fx::sequence(&[fx::fade_from_fg(theme::RED, (250, I::Linear)), fx::evolve_from(EvolveSymbolSet::Shaded, (500, I::QuadOut))])
            .with_area(body);
        self.m.add_unique_effect(Key::Body, e);
    }

    /// A request finished: the speed panel flashes bright and settles.
    pub fn finished(&mut self, speed: Rect) {
        if !ok(speed) {
            return;
        }
        let e = fx::fade_from_fg(Color::White, (900, I::QuadOut))
            .with_pattern(SweepPattern::left_to_right(12))
            .with_filter(CellFilter::Text)
            .with_area(speed);
        self.m.add_unique_effect(Key::Speed, e);
    }

    /// Reading or generating: the state badge breathes until it is idle again.
    pub fn busy(&mut self, badge: Rect) {
        if !ok(badge) {
            return;
        }
        let breathe = fx::ping_pong(fx::hsl_shift_fg([0.0, 0.0, 25.0], (700, I::SineInOut)));
        self.m.add_unique_effect(Key::Badge, fx::repeating(breathe).with_area(badge));
    }

    pub fn calm(&mut self) {
        self.m.cancel_unique_effect(Key::Badge);
    }

    /// A start, switch or restart is under way: the header shimmers until it ends (bounded).
    pub fn pending(&mut self, header: Rect) {
        if !ok(header) {
            return;
        }
        let shimmer = fx::hsl_shift_fg([40.0, 0.0, 15.0], (900, I::SineInOut))
            .with_pattern(SweepPattern::left_to_right(20))
            .with_filter(CellFilter::Text);
        let e = fx::timed_never_complete(Duration::from_secs(4).into(), fx::repeating(fx::ping_pong(shimmer)));
        self.m.add_unique_effect(Key::Header, e.with_area(header));
    }

    pub fn toast(&mut self, area: Rect, kind: ToastKind) {
        if !ok(area) {
            return;
        }
        let c = match kind {
            ToastKind::Good => theme::GREEN,
            ToastKind::Bad => theme::RED,
            ToastKind::Info => theme::CYAN,
        };
        let e = fx::parallel(&[fx::slide_in(Motion::RightToLeft, 12, 0, theme::BG, (420, I::CubicOut)), fx::fade_from_fg(c, (1200, I::QuadOut))]);
        self.m.add_unique_effect(Key::Toast, e.with_area(area));
    }

    pub fn popup(&mut self, area: Rect) {
        if !ok(area) {
            return;
        }
        let e = fx::coalesce((350, I::QuadOut)).with_pattern(RadialPattern::center().with_transition_width(4.0));
        self.m.add_unique_effect(Key::Popup, e.with_area(area));
    }
}
