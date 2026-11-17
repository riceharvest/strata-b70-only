//! Drawing: header, tabs, the five views, footer, popups.

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Padding, Bar, BarChart, BarGroup, Block, BorderType, Borders, Cell, Clear, Gauge, List, ListItem, ListState, Paragraph, Row, Sparkline, Table, TableState, Wrap},
};

use super::{App, Tab, ToastKind, theme as t};
use crate::fmt::{self, Snap};

pub fn popup_rect(screen: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(screen.width.saturating_sub(4));
    let h = h.min(screen.height.saturating_sub(2));
    Rect::new(screen.x + (screen.width - w) / 2, screen.y + (screen.height - h) / 2, w, h)
}

pub fn toast_rect(screen: Rect, text: &str) -> Rect {
    let w = (text.chars().count() as u16 + 6).min(screen.width.saturating_sub(2)).max(20);
    let h = 3;
    Rect::new(screen.right().saturating_sub(w + 1), screen.bottom().saturating_sub(h + 1), w, h)
}

fn panel(title: &str) -> Block<'_> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t::LINE))
        .title(Line::from(vec![Span::raw(" "), Span::styled(title, t::bold(t::DIM)), Span::raw(" ")]))
        .style(Style::new().bg(t::PANEL).fg(t::TEXT))
        .padding(Padding::horizontal(1))
}

fn kv<'a>(k: &'a str, v: String, c: Color) -> Line<'a> {
    Line::from(vec![Span::styled(format!("{k:<12}"), t::dim()), Span::styled(v, Style::new().fg(c))])
}

fn or_dash(v: Option<String>) -> String {
    v.unwrap_or_else(|| "—".into())
}

// ---- big digits (3 rows) ----

fn glyph(c: char) -> [&'static str; 3] {
    match c {
        '0' => ["█▀█", "█ █", "▀▀▀"],
        '1' => ["▀█ ", " █ ", "▀▀▀"],
        '2' => ["▀▀█", "█▀▀", "▀▀▀"],
        '3' => ["▀▀█", " ▀█", "▀▀▀"],
        '4' => ["█ █", "▀▀█", "  ▀"],
        '5' => ["█▀▀", "▀▀█", "▀▀▀"],
        '6' => ["█▀▀", "█▀█", "▀▀▀"],
        '7' => ["▀▀█", "  █", "  ▀"],
        '8' => ["█▀█", "█▀█", "▀▀▀"],
        '9' => ["█▀█", "▀▀█", "▀▀▀"],
        '.' => [" ", " ", "▀"],
        '-' => ["   ", "▀▀▀", "   "],
        _ => ["   ", "   ", "   "],
    }
}

fn big(text: &str, style: Style) -> Vec<Line<'static>> {
    (0..3)
        .map(|row| {
            let mut s = String::new();
            for c in text.chars() {
                s.push_str(glyph(c)[row]);
                s.push(' ');
            }
            Line::from(Span::styled(s, style))
        })
        .collect()
}

// ---- frame ----

pub fn draw(f: &mut Frame, app: &mut App) {
    let screen = f.area();
    app.areas.screen = screen;
    f.render_widget(Block::new().style(t::text()), screen);
    let [header, tabs, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Min(5), Constraint::Length(1)]).areas(screen);
    app.areas.header = header;
    app.areas.body = body;
    draw_header(f, app, header);
    draw_tabs(f, app, tabs);
    match app.tab {
        Tab::Live => {
            if app.online {
                live(f, app, body)
            } else {
                offline(f, app, body)
            }
        }
        Tab::Requests => requests(f, app, body),
        Tab::Models => models(f, app, body),
        Tab::Logs => logs(f, app, body),
        Tab::Chat => chat(f, app, body),
    }
    draw_footer(f, app, footer);
    if let Some(toast) = &app.toast {
        let r = toast_rect(screen, &toast.text);
        app.areas.toast = r;
        let c = match toast.kind {
            ToastKind::Good => t::GREEN,
            ToastKind::Bad => t::RED,
            ToastKind::Info => t::CYAN,
        };
        f.render_widget(Clear, r);
        f.render_widget(
            Paragraph::new(Line::from(vec![Span::styled("▍", Style::new().fg(c)), Span::styled(toast.text.clone(), Style::new().fg(t::TEXT))]))
                .block(Block::bordered().border_type(BorderType::Rounded).border_style(Style::new().fg(c)).style(Style::new().bg(t::PANEL))),
            r,
        );
    } else {
        app.areas.toast = toast_rect(screen, "                              ");
    }
    if let Some(c) = &app.confirm {
        let r = popup_rect(screen, 60, 8);
        app.areas.popup = r;
        f.render_widget(Clear, r);
        let mut lines: Vec<Line> = c.body.lines().map(|l| Line::from(Span::styled(l.to_string(), Style::new().fg(t::TEXT)))).collect();
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled(" y ", Style::new().fg(t::BG).bg(t::CYAN).add_modifier(Modifier::BOLD)),
            Span::styled(" yes    ", t::dim()),
            Span::styled(" any key ", Style::new().fg(t::TEXT).bg(t::LINE)),
            Span::styled(" no", t::dim()),
        ]));
        f.render_widget(
            Paragraph::new(lines).wrap(Wrap { trim: true }).block(
                Block::bordered()
                    .border_type(BorderType::Double)
                    .border_style(Style::new().fg(t::CYAN))
                    .title(Span::styled(format!(" {} ", c.title), t::bold(t::CYAN)))
                    .padding(ratatui::widgets::Padding::horizontal(1))
                    .style(Style::new().bg(t::PANEL)),
            ),
            r,
        );
    }
    if app.help {
        help(f, screen);
    }
}

fn draw_header(f: &mut Frame, app: &mut App, area: Rect) {
    let state = app.state();
    let model = app.snap.as_ref().filter(|_| app.online).map(|s| s.model.clone()).unwrap_or_else(|| "no model".into());
    let logo = " ◆ intelinference ";
    let badge = format!(" ● {state} ");
    let mut left = vec![
        Span::styled(logo, Style::new().fg(Color::White).bg(t::BLUE).add_modifier(Modifier::BOLD)),
        Span::styled("▌", Style::new().fg(t::BLUE).bg(t::BG)),
        Span::styled(format!(" {model} "), t::bold(t::TEXT)),
    ];
    let x0 = area.x + logo.chars().count() as u16 + 1 + model.chars().count() as u16 + 2;
    app.areas.badge = Rect::new(x0, area.y, badge.chars().count() as u16, 1).intersection(area);
    left.push(Span::styled(badge, t::bold(t::state_color(&state))));
    if let Some(p) = &app.pending {
        left.push(Span::styled(format!(" {p}… "), Style::new().fg(t::AMBER)));
    }
    f.render_widget(Paragraph::new(Line::from(left)), area);
    let mut right = vec![Span::styled(app.api.base().trim_start_matches("http://").to_string(), t::dim())];
    if let Some(s) = app.snap.as_ref().filter(|_| app.online) {
        if !s.version.is_empty() {
            right.push(Span::styled(format!("  engine {}", s.version), t::faint()));
        }
        if let Some(g) = &s.gpu_name {
            right.push(Span::styled(format!("  {g}"), t::faint()));
        }
        if let Some(since) = s.t_since {
            right.push(Span::styled(format!("  up {}", fmt::secs(s.time - since)), t::faint()));
        }
    }
    right.push(Span::raw(" "));
    f.render_widget(Paragraph::new(Line::from(right)).alignment(Alignment::Right), area);
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![Span::raw(" ")];
    for (i, tab) in Tab::ALL.iter().enumerate() {
        let on = *tab == app.tab;
        spans.push(Span::styled(format!("{}", i + 1), if on { t::bold(t::CYAN) } else { t::faint() }));
        spans.push(Span::styled(
            format!(" {} ", tab.title()),
            if on { Style::new().fg(t::TEXT).add_modifier(Modifier::BOLD | Modifier::UNDERLINED).underline_color(t::CYAN) } else { t::dim() },
        ));
        spans.push(Span::raw("  "));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let keys: &[(&str, &str)] = match app.tab {
        Tab::Chat => &[("enter", "send"), ("esc", "clear/back"), ("↑↓", "scroll"), ("^L", "new chat"), ("tab", "next view")],
        Tab::Models => &[("↑↓", "pick"), ("enter", "start / switch"), ("R", "re-read folder"), ("s", "stop"), ("?", "keys"), ("q", "quit")],
        Tab::Logs => &[("↑↓ pgup pgdn", "scroll"), ("G", "follow"), ("?", "keys"), ("q", "quit")],
        _ if !app.online => &[("↑↓", "pick"), ("enter", "start"), ("tab", "views"), ("?", "keys"), ("q", "quit")],
        _ => &[("l", "load"), ("u", "unload"), ("r", "restart"), ("s", "stop"), ("tab", "views"), ("?", "keys"), ("q", "quit")],
    };
    let mut spans = vec![Span::raw(" ")];
    for (k, v) in keys {
        spans.push(Span::styled(format!(" {k} "), Style::new().fg(t::TEXT).bg(t::LINE)));
        spans.push(Span::styled(format!(" {v}  "), t::dim()));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ---- Live ----

fn live(f: &mut Frame, app: &mut App, area: Rect) {
    let Some(s) = app.snap.clone() else { return };
    let tall = area.height >= 26;
    let rows = if tall {
        Layout::vertical([Constraint::Length(10), Constraint::Length(8), Constraint::Min(4)]).split(area)
    } else {
        Layout::vertical([Constraint::Length(10), Constraint::Min(6)]).split(area)
    };
    let [speed, now] = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)]).areas(rows[0]);
    app.areas.speed = speed;
    speed_panel(f, app, &s, speed);
    now_panel(f, app, &s, now);
    if tall {
        let [a, b, c] = Layout::horizontal([Constraint::Ratio(1, 3), Constraint::Ratio(1, 3), Constraint::Ratio(1, 3)]).areas(rows[1]);
        session_panel(f, &s, a);
        machine_panel(f, &s, b);
        cache_panel(f, &s, c);
        let [reqs, days] = Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).areas(rows[2]);
        recent_table(f, &s, reqs, None);
        days_chart(f, &s, days);
    } else {
        let [a, b] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(rows[1]);
        session_panel(f, &s, a);
        machine_panel(f, &s, b);
    }
}

fn speed_panel(f: &mut Frame, app: &App, s: &Snap, area: Rect) {
    let block = panel("decode");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let [num, spark] = Layout::vertical([Constraint::Length(4), Constraint::Min(1)]).areas(inner);
    let live = s.state == "generating";
    let shown = if live { app.shown_tok_s } else { s.requests.first().and_then(|r| r.decode_tok_s).unwrap_or(app.shown_tok_s) };
    let color = if live { t::CYAN } else { t::DIM };
    let [digits, side] = Layout::horizontal([Constraint::Min(14), Constraint::Length(18)]).areas(num);
    let mut lines = big(&format!("{shown:.1}"), t::bold(color));
    lines.push(Line::from(Span::styled(if live { "tok/s now" } else { "tok/s last request" }, t::dim())));
    f.render_widget(Paragraph::new(lines), digits);
    let side_lines = vec![
        kv("run mean", or_dash(s.decode_mean().map(|v| format!("{v:.1}"))), t::TEXT),
        kv("peak seen", if app.peak_tok_s > 0.0 { format!("{:.1}", app.peak_tok_s) } else { "—".into() }, t::TEXT),
        kv("drafts kept", or_dash(s.draft_pct().map(|p| format!("{p:.0}%"))), t::TEXT),
    ];
    f.render_widget(Paragraph::new(side_lines), side);
    let data: Vec<u64> = s.hist_tok_s.iter().map(|v| (*v * 10.0) as u64).collect();
    let w = spark.width as usize;
    let data = &data[data.len().saturating_sub(w)..];
    f.render_widget(Sparkline::default().data(data).style(Style::new().fg(if live { t::CYAN } else { t::BLUE })), spark);
}

fn now_panel(f: &mut Frame, app: &App, s: &Snap, area: Rect) {
    let block = panel("now");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let state = s.state.as_str();
    let mut lines = vec![Line::from(vec![
        Span::styled(state.to_uppercase(), t::bold(t::state_color(state))),
        Span::styled(s.phase.as_ref().map(|p| format!("  {p}")).unwrap_or_default(), t::dim()),
        Span::styled(if s.queued > 0.0 { format!("  {} queued", s.queued) } else { String::new() }, Style::new().fg(t::AMBER)),
    ])];
    match state {
        "reading" | "generating" => {
            let [_, g1, g2, rest] = Layout::vertical([Constraint::Length(2), Constraint::Length(2), Constraint::Length(2), Constraint::Min(0)]).areas(inner);
            if let (Some(r), Some(tot)) = (s.prompt_read, s.prompt_total) {
                let ratio = if tot > 0.0 { (r / tot).clamp(0.0, 1.0) } else { 0.0 };
                gauge(f, g1, &format!("reading the prompt  {} / {}", fmt::tokens(r), fmt::tokens(tot)), ratio, t::VIOLET);
            } else if let Some(p) = s.prompt_tokens {
                f.render_widget(Paragraph::new(kv("prompt", fmt::tokens(p), t::TEXT)), g1);
            }
            if let Some(g) = s.generated {
                let max = s.max_tokens.unwrap_or(0.0);
                let ratio = if max > 0.0 { (g / max).clamp(0.0, 1.0) } else { 0.0 };
                let label = if max > 0.0 { format!("writing  {} / {}", fmt::tokens(g), fmt::tokens(max)) } else { format!("writing  {}", fmt::tokens(g)) };
                gauge(f, g2, &label, ratio, t::CYAN);
            }
            let mut more = vec![];
            if let Some(e) = s.elapsed_s {
                more.push(kv("running for", fmt::secs(e), t::TEXT));
            }
            if let (Some(p), Some(r)) = (s.parallel, s.running) {
                more.push(kv("slots", format!("{r} of {p} busy"), t::TEXT));
            }
            f.render_widget(Paragraph::new(lines), inner);
            f.render_widget(Paragraph::new(more), rest);
            return;
        }
        "unloaded" => {
            lines.push(Line::default());
            lines.push(Line::from(Span::styled("The model is unloaded: the GPU is free for other work.", t::dim())));
            lines.push(Line::from(vec![Span::styled("l", t::bold(t::CYAN)), Span::styled(" loads it now; the next request loads it too.", t::dim())]));
        }
        _ => {
            if let Some(r) = s.requests.first() {
                lines.push(Line::default());
                lines.push(Line::from(Span::styled(format!("last request at {}", fmt::clock(r.time)), t::dim())));
                lines.push(kv("prompt", format!("{}  ({} reused, {} read)", fmt::tokens(r.prompt), or_dash(fmt::pct(r.reused, r.prompt).map(|p| format!("{p:.0}%"))), fmt::tokens(r.read)), t::TEXT));
                lines.push(kv("first token", or_dash(r.first_token_s.map(fmt::secs)), t::TEXT));
                lines.push(kv("answer", format!("{} tokens in {}", fmt::tokens(r.output), fmt::secs(r.duration_s)), t::TEXT));
                lines.push(kv("drafts kept", or_dash(fmt::pct(r.drafts_accepted, r.drafts_offered).map(|p| format!("{p:.0}%"))), t::TEXT));
            } else {
                lines.push(Line::from(Span::styled("No requests yet.", t::dim())));
            }
        }
    }
    let _ = app;
    f.render_widget(Paragraph::new(lines), inner);
}

fn gauge(f: &mut Frame, area: Rect, label: &str, ratio: f64, c: Color) {
    let [l, g] = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);
    f.render_widget(Paragraph::new(Span::styled(label.to_string(), t::dim())), l);
    f.render_widget(Gauge::default().ratio(ratio).label("").gauge_style(Style::new().fg(c).bg(t::LINE)).use_unicode(true), g);
}

fn session_panel(f: &mut Frame, s: &Snap, area: Rect) {
    let lines = vec![
        kv("requests", format!("{}", s.t_requests), t::TEXT),
        kv("prompt", format!("{}  ({} reused)", fmt::tokens(s.t_prompt), or_dash(s.reuse_pct().map(|p| format!("{p:.1}%")))), t::TEXT),
        kv("answers", fmt::tokens(s.t_output), t::TEXT),
        kv("read speed", or_dash(s.read_mean().map(|v| format!("{v:.0} tok/s"))), t::TEXT),
        kv("context", format!("{}  kv {}", or_dash(s.context.map(fmt::tokens)), s.kv.as_deref().unwrap_or("—")), t::DIM),
        kv("all time", format!("{} requests, {} written", fmt::tokens(s.life_requests), fmt::tokens(s.life_output)), t::DIM),
        kv("drafting", format!("{}{}", s.drafting.as_deref().unwrap_or("—"), s.spec.map(|x| format!("  up to {x}")).unwrap_or_default()), t::DIM),
    ];
    f.render_widget(Paragraph::new(lines).block(panel("this run")), area);
}

fn machine_panel(f: &mut Frame, s: &Snap, area: Rect) {
    let block = panel("machine");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let [ram, vram, cpu] = Layout::vertical([Constraint::Length(2), Constraint::Length(2), Constraint::Min(1)]).areas(inner);
    if let (Some(u), Some(tot)) = (s.ram_used, s.ram_total) {
        let ratio = (u / tot).clamp(0.0, 1.0);
        let c = if ratio > 0.9 { t::RED } else if ratio > 0.75 { t::AMBER } else { t::GREEN };
        gauge(f, ram, &format!("RAM  {} of {}", fmt::gib(u), fmt::gib(tot)), ratio, c);
    }
    let mut v = vec![];
    if let Some(m) = s.vram_free_mib {
        v.push(Span::styled("VRAM free ", t::dim()));
        v.push(Span::styled(fmt::mib(m), Style::new().fg(if m < 1024.0 { t::AMBER } else { t::TEXT })));
    }
    if let Some(e) = s.expert_slots {
        v.push(Span::styled(format!("   experts {}", fmt::tokens(e)), t::dim()));
    }
    f.render_widget(Paragraph::new(Line::from(v)), vram);
    let data: Vec<u64> = s.hist_cpu.iter().map(|v| *v as u64).collect();
    let [l, sp] = Layout::horizontal([Constraint::Length(9), Constraint::Min(1)]).areas(cpu);
    f.render_widget(Paragraph::new(Span::styled(format!("CPU {:>3.0}%", s.cpu.unwrap_or(0.0)), t::dim())), l);
    let w = sp.width as usize;
    f.render_widget(Sparkline::default().data(&data[data.len().saturating_sub(w)..]).max(100).style(Style::new().fg(t::VIOLET)), sp);
}

fn cache_panel(f: &mut Frame, s: &Snap, area: Rect) {
    let mut lines = vec![kv("parked", format!("{} parks, {} back", s.cc_parks, s.cc_restores), t::TEXT)];
    if let Some(files) = s.cc_ssd_files {
        lines.push(kv("on SSD", format!("{} files, {}", files, s.cc_ssd_gib.map(|g| format!("{g:.1} GiB")).unwrap_or_default()), t::TEXT));
        if let Some(r) = s.cc_ssd_restores {
            lines.push(kv("from SSD", format!("{r} read back"), t::TEXT));
        }
    }
    if let Some(p) = s.return_gap_p50 {
        lines.push(kv("agent gap", format!("{} median", fmt::secs(p)), t::DIM));
    }
    for th in s.threads.iter().take(3) {
        let key: String = th.key.chars().rev().take(6).collect::<String>().chars().rev().collect();
        lines.push(Line::from(vec![
            Span::styled(if th.pinned { "◆ " } else { "· " }, Style::new().fg(t::CYAN)),
            Span::styled(format!("…{key} "), t::faint()),
            Span::styled(format!("{} ", th.state), Style::new().fg(t::TEXT)),
            Span::styled(format!("{} tok, {} turns", fmt::tokens(th.tokens), th.turns), t::dim()),
        ]));
    }
    f.render_widget(Paragraph::new(lines).block(panel("conversations")), area);
}

fn recent_table(f: &mut Frame, s: &Snap, area: Rect, state: Option<&mut TableState>) {
    let header = Row::new(["time", "prompt", "reused", "1st token", "answer", "tok/s", "drafts", "end"].map(|h| Cell::from(Span::styled(h, t::bold(t::DIM)))));
    let rows: Vec<Row> = s
        .requests
        .iter()
        .map(|r| {
            let tps = r.decode_tok_s.unwrap_or(0.0);
            let c = if tps >= 100.0 { t::CYAN } else if tps >= 50.0 { t::TEXT } else { t::AMBER };
            Row::new(vec![
                Cell::from(Span::styled(fmt::clock(r.time), t::dim())),
                Cell::from(fmt::tokens(r.prompt)),
                Cell::from(or_dash(fmt::pct(r.reused, r.prompt).map(|p| format!("{p:.0}%")))),
                Cell::from(or_dash(r.first_token_s.map(fmt::secs))),
                Cell::from(fmt::tokens(r.output)),
                Cell::from(Span::styled(r.decode_tok_s.map(|v| format!("{v:.1}")).unwrap_or("—".into()), Style::new().fg(c))),
                Cell::from(or_dash(fmt::pct(r.drafts_accepted, r.drafts_offered).map(|p| format!("{p:.0}%")))),
                Cell::from(Span::styled(r.finish.clone(), if r.finish == "stop" || r.finish == "tool_calls" { t::dim() } else { Style::new().fg(t::AMBER) })),
            ])
        })
        .collect();
    let widths = [8, 7, 7, 9, 7, 7, 7, 10].map(Constraint::Length);
    let table = Table::new(rows, widths)
        .header(header)
        .column_spacing(2)
        .row_highlight_style(Style::new().bg(t::LINE).fg(Color::White))
        .block(panel("requests"));
    match state {
        Some(st) => f.render_stateful_widget(table, area, st),
        None => f.render_widget(table, area),
    }
}

fn days_chart(f: &mut Frame, s: &Snap, area: Rect) {
    let block = panel("tokens written per day");
    let inner = block.inner(area);
    f.render_widget(block, area);
    if s.days.is_empty() {
        return;
    }
    let bar_w = 5u16;
    let fit = (inner.width / (bar_w + 1)).max(1) as usize;
    let days = &s.days[s.days.len().saturating_sub(fit)..];
    let bars: Vec<Bar> = days
        .iter()
        .map(|d| {
            Bar::default()
                .value(d.output as u64)
                .text_value(fmt::tokens(d.output))
                .label(Line::from(d.date.get(5..).unwrap_or(&d.date).to_string()))
                .style(Style::new().fg(t::BLUE))
                .value_style(Style::new().fg(t::TEXT).bg(t::BLUE))
        })
        .collect();
    f.render_widget(BarChart::default().data(BarGroup::default().bars(&bars)).bar_width(bar_w).bar_gap(1).label_style(t::dim()), inner);
}

// ---- offline ----

const LOGO: [&str; 3] = ["▀█▀ █▄ █ ▀█▀ █▀▀ █   ▀█▀ █▄ █ █▀▀ █▀▀ █▀█ █▀▀ █▄ █ █▀▀ █▀▀", " █  █ ▀█  █  █▀▀ █    █  █ ▀█ █▀  █▀▀ █▀▄ █▀▀ █ ▀█ █   █▀▀", "▀▀▀ ▀  ▀  ▀  ▀▀▀ ▀▀▀ ▀▀▀ ▀  ▀ ▀   ▀▀▀ ▀ ▀ ▀▀▀ ▀  ▀ ▀▀▀ ▀▀▀"];

fn offline(f: &mut Frame, app: &mut App, area: Rect) {
    let starting = app.pending.is_some();
    let card = popup_rect(area, 72, (app.configs.len() as u16 + 14).min(area.height));
    let mut lines: Vec<Line> = LOGO.iter().map(|l| Line::from(Span::styled(*l, t::bold(if starting { t::CYAN } else { t::BLUE })))).collect();
    lines.push(Line::from(Span::styled("Strata on the Intel Arc B70", t::dim())));
    lines.push(Line::default());
    if starting {
        lines.push(Line::from(Span::styled(format!("{}…", app.pending.as_deref().unwrap_or("")), t::bold(t::AMBER))));
        if let Some(last) = app.log_lines.last() {
            let l: String = last.chars().take(66).collect();
            lines.push(Line::from(Span::styled(l, t::faint())));
        }
    } else {
        lines.push(Line::from(vec![Span::styled("nothing on ", t::dim()), Span::styled(app.api.base().to_string(), Style::new().fg(t::TEXT))]));
        lines.push(Line::from(Span::styled("pick a run config and press enter to start it", t::dim())));
    }
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), card);
    let list_area = Rect::new(card.x + 6, card.y + 9, card.width.saturating_sub(12), card.height.saturating_sub(9));
    config_list(f, app, list_area, false);
}

// ---- Models ----

fn config_list(f: &mut Frame, app: &App, area: Rect, bordered: bool) {
    if app.configs.is_empty() {
        f.render_widget(Paragraph::new(Span::styled(format!("no run configs in {}", app.s.config_dir.display()), t::dim())).wrap(Wrap { trim: true }), area);
        return;
    }
    let items: Vec<ListItem> = app
        .configs
        .iter()
        .map(|c| {
            let on = app.online && app.current_config.as_deref() == Some(c.file.as_str());
            let dot = if on { Span::styled("● ", Style::new().fg(if app.loaded { t::GREEN } else { t::AMBER })) } else { Span::raw("  ") };
            let name = c.file.trim_end_matches(".json").to_string();
            let mut extra = vec![c.model.clone()];
            if let Some(x) = c.context {
                extra.push(format!("ctx {}", fmt::tokens(x as f64)));
            }
            ListItem::new(Line::from(vec![dot, Span::styled(format!("{name:<22}"), t::bold(t::TEXT)), Span::styled(extra.join("  "), t::dim())]))
        })
        .collect();
    let mut st = ListState::default().with_selected(Some(app.sel_config.min(app.configs.len() - 1)));
    let mut list = List::new(items).highlight_style(Style::new().bg(t::LINE).fg(Color::White)).highlight_symbol("› ");
    if bordered {
        list = list.block(panel("run configs"));
    }
    f.render_stateful_widget(list, area, &mut st);
}

fn models(f: &mut Frame, app: &mut App, area: Rect) {
    let [list, detail] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area);
    config_list(f, app, list, true);
    let mut lines = vec![Line::from(Span::styled(app.s.config_dir.display().to_string(), t::faint())), Line::default()];
    if let Some(c) = app.configs.get(app.sel_config) {
        let running = app.online && app.current_config.as_deref() == Some(c.file.as_str());
        lines.push(Line::from(Span::styled(c.file.clone(), t::bold(t::CYAN))));
        lines.push(kv("model", c.model.clone(), t::TEXT));
        lines.push(kv("weights", or_dash(c.weights.clone()), t::TEXT));
        lines.push(kv("family", or_dash(c.arch.clone()), t::TEXT));
        lines.push(kv("context", or_dash(c.context.map(|x| fmt::tokens(x as f64))), t::TEXT));
        lines.push(kv("kv cache", or_dash(c.kv.clone()), t::TEXT));
        lines.push(kv("batch", or_dash(c.batch.map(|b| b.to_string())), t::TEXT));
        lines.push(Line::default());
        let hint = if running {
            "running now".to_string()
        } else if !app.online {
            "enter: start the server with this config".to_string()
        } else if app.switchable.iter().any(|x| x == &c.file) {
            "enter: switch the server to this config (it restarts)".to_string()
        } else {
            "not in the running server's folder: stop it first (s), then start this one".to_string()
        };
        lines.push(Line::from(Span::styled(hint, Style::new().fg(if running { t::GREEN } else { t::DIM }))));
    }
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(panel("details")), detail);
}

// ---- Requests ----

fn requests(f: &mut Frame, app: &mut App, area: Rect) {
    let Some(s) = app.snap.clone().filter(|_| app.online) else {
        f.render_widget(Paragraph::new(Span::styled("no server", t::dim())).block(panel("requests")), area);
        return;
    };
    if s.requests.is_empty() {
        f.render_widget(Paragraph::new(Span::styled("no requests yet", t::dim())).block(panel("requests")), area);
        return;
    }
    app.sel_request = app.sel_request.min(s.requests.len() - 1);
    let [table, detail] = Layout::vertical([Constraint::Min(5), Constraint::Length(4)]).areas(area);
    let mut st = TableState::default().with_selected(Some(app.sel_request));
    recent_table(f, &s, table, Some(&mut st));
    let r = &s.requests[app.sel_request];
    let read_speed = r.prompt_ms.filter(|m| *m > 0.0).map(|m| r.read / m * 1000.0);
    let lines = vec![
        Line::from(vec![
            Span::styled(format!("{}  ", fmt::clock(r.time)), t::dim()),
            Span::styled(format!("{} prompt tokens: {} reused, {} read", fmt::tokens(r.prompt), fmt::tokens(r.reused), fmt::tokens(r.read)), Style::new().fg(t::TEXT)),
            Span::styled(read_speed.map(|v| format!(" at {v:.0} tok/s")).unwrap_or_default(), t::dim()),
        ]),
        Line::from(vec![
            Span::styled(format!("answer {} tokens", fmt::tokens(r.output)), Style::new().fg(t::TEXT)),
            Span::styled(format!("  in {}", fmt::secs(r.duration_s)), t::dim()),
            Span::styled(format!("  drafts {} of {} kept", r.drafts_accepted, r.drafts_offered), t::dim()),
            Span::styled(format!("  ended: {}", r.finish), t::dim()),
        ]),
    ];
    f.render_widget(Paragraph::new(lines).block(panel("selected")), detail);
}

// ---- Logs ----

fn log_style(l: &str) -> Style {
    let low = l.to_ascii_lowercase();
    if low.contains("error") || low.contains("failed") || low.contains("traceback") {
        Style::new().fg(t::RED)
    } else if low.contains("warn") || low.contains("refused") || low.contains("skip") {
        Style::new().fg(t::AMBER)
    } else if l.starts_with("[intelinference]") {
        t::bold(t::CYAN)
    } else if low.contains("ready") || low.contains("loaded") {
        Style::new().fg(t::GREEN)
    } else {
        Style::new().fg(t::TEXT)
    }
}

fn logs(f: &mut Frame, app: &mut App, area: Rect) {
    let title = format!("{}{}", app.s.log.display(), if app.log_scroll > 0 { format!("  (scrolled {} up)", app.log_scroll) } else { String::new() });
    let block = panel(&title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if app.log_lines.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("no log yet (servers started here write to it; point --log at another server's log)", t::dim())).wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }
    let h = inner.height as usize;
    app.log_scroll = app.log_scroll.min(app.log_lines.len().saturating_sub(h));
    let end = app.log_lines.len() - app.log_scroll;
    let start = end.saturating_sub(h);
    let lines: Vec<Line> = app.log_lines[start..end].iter().map(|l| Line::from(Span::styled(l.clone(), log_style(l)))).collect();
    f.render_widget(Paragraph::new(lines), inner);
}

// ---- Chat ----

fn chat(f: &mut Frame, app: &mut App, area: Rect) {
    let [msgs, input] = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).areas(area);
    let mut text = Text::default();
    if app.chat.is_empty() {
        text.lines.push(Line::from(Span::styled("Ask the model something. Thinking shows dimmed; enter sends.", t::dim())));
    }
    for m in &app.chat {
        let (who, c) = if m.role == "user" { ("you", t::CYAN) } else { ("model", t::GREEN) };
        text.lines.push(Line::from(Span::styled(who, t::bold(c))));
        if !m.reasoning.is_empty() {
            for l in m.reasoning.lines() {
                text.lines.push(Line::from(Span::styled(format!("  {l}"), Style::new().fg(t::FAINT).italic())));
            }
        }
        for l in m.text.lines() {
            text.lines.push(Line::from(Span::styled(format!("  {l}"), Style::new().fg(t::TEXT))));
        }
        text.lines.push(Line::default());
    }
    let block = panel("chat");
    let inner = block.inner(msgs);
    // keep the end in view: wrap-aware line count
    let width = inner.width.max(1) as usize;
    let total: usize = text.lines.iter().map(|l| (l.width().max(1) + width - 1) / width).sum();
    let over = total.saturating_sub(inner.height as usize) as u16;
    app.chat_scroll = app.chat_scroll.min(over);
    let scroll = over - app.chat_scroll;
    f.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }).scroll((scroll, 0)).block(block), msgs);
    let cursor = if app.chat_busy { Span::styled(" ⋯", Style::new().fg(t::CYAN)) } else { Span::styled("▏", Style::new().fg(t::CYAN).add_modifier(Modifier::SLOW_BLINK)) };
    f.render_widget(
        Paragraph::new(Line::from(vec![Span::styled("› ", t::bold(t::CYAN)), Span::styled(app.input.clone(), Style::new().fg(t::TEXT)), cursor]))
            .block(Block::bordered().border_type(BorderType::Rounded).border_style(Style::new().fg(if app.chat_busy { t::LINE } else { t::BLUE })).style(Style::new().bg(t::PANEL))),
        input,
    );
}

// ---- help ----

fn help(f: &mut Frame, screen: Rect) {
    let r = popup_rect(screen, 58, 18);
    f.render_widget(Clear, r);
    let rows = [
        ("1-5, tab", "switch view"),
        ("enter", "start / switch to the picked run config"),
        ("l / u", "load the model / unload it (frees the GPU)"),
        ("r", "restart the server (the run config is read again)"),
        ("s", "stop the server"),
        ("R", "read the run config folder again"),
        ("↑↓ j k", "move, scroll"),
        ("G", "logs: follow the end"),
        ("q, esc", "quit (the server keeps running)"),
        ("", ""),
        ("intelinference --help", "the one-shot commands"),
    ];
    let lines: Vec<Line> = rows.iter().map(|(k, v)| Line::from(vec![Span::styled(format!("{k:>22}  "), t::bold(t::CYAN)), Span::styled(*v, Style::new().fg(t::TEXT))])).collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::bordered().border_type(BorderType::Rounded).border_style(Style::new().fg(t::BLUE)).title(Span::styled(" keys ", t::bold(t::CYAN))).style(Style::new().bg(t::PANEL)),
        ),
        r,
    );
}
