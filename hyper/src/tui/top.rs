use std::future::Future;

use anyhow::Result;
use crossterm::event::Event;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use crossterm::event::{self};
use hyperactor::ActorId;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use ratatui::DefaultTerminal;
use ratatui::Frame;
use ratatui::layout::Constraint;
use ratatui::layout::Direction;
use ratatui::layout::Layout;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Cell;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Row;
use ratatui::widgets::Table;
use ratatui::widgets::Wrap;
use regex::Regex;

/// Frequency of data fetch
const REFRESH_RATE: tokio::time::Duration = tokio::time::Duration::from_millis(5000);
/// Text displayed in the filter area when the filter is empty
const EMPTY_FILTER_TEXT: &str = "enter a pattern to filter | :q to quit";

trait Component {
    type Props;
    type State;
    fn render(&mut self, f: &mut Frame, area: Rect, props: Self::Props);
    fn handle(&mut self, event: &KeyEvent);
}

#[derive(Clone)]
pub struct ActorInfo {
    pub actor_id: String,
    pub message_count: usize,
}

struct AppProps {
    actors: Vec<ActorInfo>,
}
struct AppState {
    filter: String,
    execution_id: String,
    cursor_position: usize,
}

pub struct App {
    state: AppState,
    table: AppTable,
    filter: AppFilter,
    exit: bool,
}

impl App {
    pub fn new(execution_id: String) -> Self {
        Self {
            state: AppState {
                filter: String::new(),
                execution_id,
                cursor_position: 0,
            },
            table: AppTable::new(),
            filter: AppFilter::new(),
            exit: false,
        }
    }

    #[cfg(test)]
    fn set_cursor_position_unit_tests_only(&mut self, cursor_position: usize) {
        self.state.cursor_position = cursor_position;
    }

    pub async fn run<F, Fut>(
        &mut self,
        terminal: &mut DefaultTerminal,
        fetch_actors: F,
    ) -> Result<()>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Vec<ActorInfo>>>,
    {
        let mut last_fetched_data = RealClock.now();
        let mut actors = vec![];

        while !self.exit {
            if last_fetched_data.elapsed() > REFRESH_RATE {
                if let Ok(res) = fetch_actors().await {
                    actors = res;
                }
                last_fetched_data = RealClock.now();
            }

            terminal.draw(|f| {
                self.render(
                    f,
                    f.area(),
                    AppProps {
                        actors: actors.clone(),
                    },
                )
            })?;

            if let Event::Key(key_event) = event::read()? {
                self.handle(&key_event);
            }
        }
        Ok(())
    }
}

impl Component for App {
    type Props = AppProps;
    type State = AppState;

    fn render(&mut self, f: &mut Frame, area: Rect, props: AppProps) {
        let width = area.width as usize;
        let filter_query_lines = std::cmp::min(
            self.state.filter.len() / width + 1,
            area.height as usize - 2,
        );

        let rects = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(0),
                Constraint::Length(1 + filter_query_lines as u16),
            ])
            .split(area);

        f.render_widget(
            Paragraph::new(Span::styled(
                format!("Execution id: {}", self.state.execution_id),
                Style::default().fg(Color::Cyan),
            )),
            rects[0],
        );

        let re = if self.state.filter.is_empty() {
            None
        } else {
            Regex::new(&self.state.filter).ok()
        };

        let total_actors = props.actors.len();
        let actors_matching_filter = props
            .actors
            .into_iter()
            .filter(|actor_info| {
                let actor_id_string = actor_info.actor_id.to_string();
                if let Some(re) = &re {
                    re.is_match(&actor_id_string)
                } else {
                    actor_id_string
                        .to_lowercase()
                        .contains(&self.state.filter.to_lowercase())
                }
            })
            .collect::<Vec<_>>();
        let matching_actors = actors_matching_filter.len();

        self.table.render(
            f,
            rects[1],
            AppTableProps {
                actors: actors_matching_filter,
            },
        );

        self.filter.render(
            f,
            rects[2],
            AppFilterProps {
                query: self.state.filter.clone(),
                total_actors,
                matching_actors,
                cursor_position: self.state.cursor_position,
            },
        );
    }

    fn handle(&mut self, event: &KeyEvent) {
        match event.code {
            KeyCode::Char(c) if event.modifiers.contains(KeyModifiers::CONTROL) => match c {
                'a' => {
                    self.state.cursor_position = 0;
                }
                'e' => {
                    self.state.cursor_position = self.state.filter.len();
                }
                _ => {}
            },
            KeyCode::Enter if self.state.filter == ":q" => {
                self.exit = true;
            }
            KeyCode::Char(c) => {
                if self.state.cursor_position == self.state.filter.len() {
                    self.state.filter.push(c);
                } else {
                    self.state.filter.insert(self.state.cursor_position, c);
                }
                self.state.cursor_position =
                    std::cmp::min(self.state.filter.len(), self.state.cursor_position + 1);
            }
            KeyCode::Backspace => {
                if self.state.cursor_position > 0 {
                    self.state.filter.remove(self.state.cursor_position - 1);
                    self.state.cursor_position -= 1;
                }
            }
            KeyCode::Esc => {
                self.state.cursor_position = 0;
                self.state.filter.clear();
            }
            KeyCode::Left => {
                if self.state.cursor_position > 0 {
                    self.state.cursor_position -= 1;
                }
            }
            KeyCode::Right => {
                self.state.cursor_position =
                    std::cmp::min(self.state.filter.len(), self.state.cursor_position + 1);
            }
            _ => {}
        }
        self.filter.handle(event);
        self.table.handle(event);
    }
}

struct AppFilterProps {
    query: String,
    total_actors: usize,
    matching_actors: usize,
    cursor_position: usize,
}

struct AppFilter {}

impl AppFilter {
    fn new() -> Self {
        Self {}
    }
}

impl Component for AppFilter {
    type Props = AppFilterProps;
    type State = ();

    fn render(&mut self, f: &mut Frame, area: Rect, props: Self::Props) {
        let width = area.width as usize;
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(area);

        let label_string = format!("{}/{}", props.matching_actors, props.total_actors);
        let dash_count = width - label_string.len();
        let line_string = format!("{label_string}{}", "-".repeat(dash_count));

        f.render_widget(Paragraph::new(Span::raw(line_string)), layout[0]);

        if props.query.is_empty() {
            f.render_widget(
                Paragraph::new(EMPTY_FILTER_TEXT).style(Style::default().fg(Color::DarkGray)),
                layout[1],
            );
            return;
        }

        let (before_cursor, after_cursor) = props
            .query
            .split_at(props.cursor_position.min(props.query.len()));
        let highlighted_char = after_cursor.chars().next();

        let mut spans = vec![Span::raw(before_cursor)];

        if let Some(c) = highlighted_char {
            spans.push(Span::styled(
                c.to_string(),
                Style::default().add_modifier(Modifier::REVERSED),
            ));
            spans.push(Span::raw(&after_cursor[c.len_utf8()..]));
        } else {
            spans.push(Span::styled(
                " ",
                Style::default().add_modifier(Modifier::REVERSED),
            ));
        }

        f.render_widget(
            Paragraph::new(Line::from(spans)).wrap(Wrap { trim: false }),
            layout[1],
        );
    }

    fn handle(&mut self, _event: &KeyEvent) {}
}

struct AppTableProps {
    actors: Vec<ActorInfo>,
}

struct AppTableState {
    scroll_offset: usize,
}

struct AppTable {
    state: AppTableState,
}

impl AppTable {
    fn new() -> Self {
        Self {
            state: AppTableState { scroll_offset: 0 },
        }
    }
}

impl Component for AppTable {
    type Props = AppTableProps;
    type State = AppTableState;

    fn render(&mut self, f: &mut Frame, area: Rect, props: AppTableProps) {
        let titles_and_widths = vec![
            ("Actor", Constraint::Length(30)),
            ("World", Constraint::Length(15)),
            ("Message Count", Constraint::Length(20)),
        ];

        let header = Row::new(
            titles_and_widths
                .iter()
                .map(|(title, _)| Cell::from(*title)),
        )
        .style(Style::default().add_modifier(Modifier::BOLD));

        // Calculate the maximum number of rows that can be displayed
        let max_rows = area.height.saturating_sub(1) as usize; // Subtract 1 for the header row

        // Ensure scroll offset is within bounds
        self.state.scroll_offset = self
            .state
            .scroll_offset
            .min(props.actors.len().saturating_sub(max_rows))
            .max(0);

        // Get the visible slice of actors based on scroll offset
        let visible_actors = props
            .actors
            .into_iter()
            .skip(self.state.scroll_offset)
            .take(max_rows);

        let rows = visible_actors.map(|actor| {
            let parsed_actor_id = actor.actor_id.parse::<ActorId>();
            let world_name = parsed_actor_id
                .map(|id| id.world_name().to_string())
                .unwrap_or("unknown".to_string());

            Row::new(vec![
                actor.actor_id.to_string(),
                world_name,
                actor.message_count.to_string(),
            ])
        });

        let table = Table::new(
            rows,
            titles_and_widths
                .into_iter()
                .map(|(_, c)| c)
                .collect::<Vec<_>>(),
        )
        .header(header);

        f.render_widget(table, area);
    }

    fn handle(&mut self, event: &KeyEvent) {
        match event.code {
            KeyCode::Up => {
                if self.state.scroll_offset > 0 {
                    self.state.scroll_offset -= 1;
                }
            }
            KeyCode::Down => {
                self.state.scroll_offset += 1;
            }
            KeyCode::PageUp => {
                self.state.scroll_offset = self.state.scroll_offset.saturating_sub(10);
            }
            KeyCode::PageDown => {
                self.state.scroll_offset += 10;
            }
            KeyCode::Home => {
                self.state.scroll_offset = 0;
            }
            KeyCode::End => {
                self.state.scroll_offset = usize::MAX;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyEventKind;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    use super::*;

    #[test]
    fn test_app_table_render_empty() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an empty AppTable
        let mut app_table = AppTable::new();

        // Create empty props
        let props = AppTableProps { actors: vec![] };

        // Render the table
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app_table.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("{}", buffer_content);

        // Verify the table headers are rendered
        assert!(buffer_content.contains("Actor"));
        assert!(buffer_content.contains("World"));
        assert!(buffer_content.contains("Message Count"));
    }

    #[test]
    fn test_app_table_render_with_actors() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an empty AppTable
        let mut app_table = AppTable::new();

        // Create props with some actors
        let props = AppTableProps {
            actors: vec![
                ActorInfo {
                    actor_id: "test.actor1".to_string(),
                    message_count: 10,
                },
                ActorInfo {
                    actor_id: "test.actor2".to_string(),
                    message_count: 20,
                },
            ],
        };

        // Render the table
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app_table.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("{}", buffer_content);

        // Verify the table headers are rendered
        assert!(buffer_content.contains("Actor"));
        assert!(buffer_content.contains("World"));
        assert!(buffer_content.contains("Message Count"));

        // Verify the actor data is rendered
        assert!(buffer_content.contains("test.actor1"));
        assert!(buffer_content.contains("test.actor2"));
        assert!(buffer_content.contains("10"));
        assert!(buffer_content.contains("20"));
    }

    #[test]
    fn test_app_table_filter() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an empty AppTable
        let mut app = App::new("execution id".to_string());

        // Create props with some actors and a filter
        let props = AppProps {
            actors: vec![
                ActorInfo {
                    actor_id: "test.actor1".to_string(),
                    message_count: 10,
                },
                ActorInfo {
                    actor_id: "test.actor2".to_string(),
                    message_count: 20,
                },
                ActorInfo {
                    actor_id: "other.actor3".to_string(),
                    message_count: 30,
                },
            ],
        };
        for char in "test".chars() {
            app.handle(&KeyCode::Char(char).into());
        }

        // Render the table
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("{}", buffer_content);

        // Verify the table headers are rendered
        assert!(buffer_content.contains("Actor"));
        assert!(buffer_content.contains("World"));
        assert!(buffer_content.contains("Message Count"));

        // Verify the filtered actor data is rendered
        assert!(buffer_content.contains("test.actor1"));
        assert!(buffer_content.contains("test.actor2"));
        assert!(buffer_content.contains("10"));
        assert!(buffer_content.contains("20"));

        // Verify the filtered out actor is not rendered
        assert!(!buffer_content.contains("other.actor3"));
        assert!(!buffer_content.contains("30"));
    }

    #[test]
    fn test_app_table_regex_filter() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an empty AppTable
        let mut app = App::new("execution id".to_string());

        // Create props with some actors and a regex filter
        let props = AppProps {
            actors: vec![
                ActorInfo {
                    actor_id: "test.actor1".to_string(),
                    message_count: 10,
                },
                ActorInfo {
                    actor_id: "test.actor2".to_string(),
                    message_count: 20,
                },
                ActorInfo {
                    actor_id: "other.actor3".to_string(),
                    message_count: 30,
                },
            ],
        };

        for char in "actor[12]".chars() {
            app.handle(&KeyCode::Char(char).into());
        }

        // Render the table
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("{}", buffer_content);

        // Verify the table headers are rendered
        assert!(buffer_content.contains("Actor"));
        assert!(buffer_content.contains("World"));
        assert!(buffer_content.contains("Message Count"));

        // Verify the filtered actor data is rendered
        assert!(buffer_content.contains("test.actor1"));
        assert!(buffer_content.contains("test.actor2"));
        assert!(buffer_content.contains("10"));
        assert!(buffer_content.contains("20"));

        // Verify the filtered out actor is not rendered
        assert!(!buffer_content.contains("other.actor3"));
        assert!(!buffer_content.contains("30"));
    }

    #[test]
    fn test_app_filter_render_empty() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 3);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an AppFilter
        let mut app_filter = AppFilter::new();

        // Create empty props
        let props = AppFilterProps {
            query: String::new(),
            total_actors: 2,
            matching_actors: 2,
            cursor_position: 0,
        };

        // Render the filter
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 3);
                app_filter.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("{}", buffer_content);

        // Verify the filter title is displayed
        assert!(buffer_content.contains("2/2"));

        // Verify the empty filter text is displayed
        assert!(buffer_content.contains(EMPTY_FILTER_TEXT));
    }

    #[test]
    fn test_app_filter_render_with_text() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 3);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an AppFilter
        let mut app_filter = AppFilter::new();

        // Create props with filter text
        let props = AppFilterProps {
            query: "test".to_string(),
            total_actors: 2,
            matching_actors: 2,
            cursor_position: 0,
        };

        // Render the filter
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 3);
                app_filter.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("{}", buffer_content);

        // Verify the filter title is displayed
        assert!(buffer_content.contains("2/2"));

        // Verify the filter text is displayed
        assert!(buffer_content.contains("test"));
    }

    #[test]
    fn test_app_filter_render_with_long_text() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 5);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an AppFilter
        let mut app_filter = AppFilter::new();

        // Create props with a long filter text
        let mut long_text = String::from("This is a ");
        for _ in 0..20 {
            long_text.push_str("very ");
        }
        long_text += "long filter text that might exceed the display area width and test how the component handles it";

        let props = AppFilterProps {
            query: long_text.to_string(),
            total_actors: 2,
            matching_actors: 2,
            cursor_position: 0,
        };

        // Render the filter
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 5);
                app_filter.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("{}", buffer_content);

        // Verify the filter title is displayed
        assert!(buffer_content.contains("2/2"));
        // Verify at least part of the long text is displayed
        // We don't check for the entire text as it might be truncated or wrapped
        assert!(buffer_content.contains("handles it"));
    }

    #[test]
    fn test_cursor_at_zero_backspace() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with empty filter
        let mut app = App::new("test_execution_id".to_string());

        // Set cursor position to 0
        app.set_cursor_position_unit_tests_only(0);

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        let initial_buffer = terminal.backend().buffer().clone();

        // Send backspace keystroke
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Backspace,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, AppProps { actors });
            })
            .unwrap();

        let final_buffer = terminal.backend().buffer().clone();

        // The cursor should still be at position 0 and the filter should remain empty
        // So the buffers should be identical
        assert_eq!(initial_buffer, final_buffer);
    }
    #[test]
    fn test_cursor_at_zero_left_key() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with empty filter
        let mut app = App::new("test_execution_id".to_string());

        // Set cursor position to 0
        app.set_cursor_position_unit_tests_only(0);

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        let initial_buffer = terminal.backend().buffer().clone();

        // Send left key keystroke
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Left,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, AppProps { actors });
            })
            .unwrap();

        let final_buffer = terminal.backend().buffer().clone();

        // The cursor should still be at position 0
        // So the buffers should be identical
        assert_eq!(initial_buffer, final_buffer);
    }

    #[test]
    fn test_cursor_at_end_right_key() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with some text in filter
        let mut app = App::new("test_execution_id".to_string());

        // Add some text to the filter
        for c in "test".chars() {
            app.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Set cursor position to end
        app.set_cursor_position_unit_tests_only(4); // "test" has 4 characters

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        let initial_buffer = terminal.backend().buffer().clone();

        // Send right key keystroke
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Right,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, AppProps { actors });
            })
            .unwrap();

        let final_buffer = terminal.backend().buffer().clone();

        // The cursor should still be at the end
        // So the buffers should be identical
        assert_eq!(initial_buffer, final_buffer);
    }

    #[test]
    fn test_cursor_at_end_type() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with some text in filter
        let mut app = App::new("test_execution_id".to_string());

        // Add some text to the filter
        for c in "test".chars() {
            app.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Set cursor position to end
        app.set_cursor_position_unit_tests_only(4); // "test" has 4 characters

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Send a keystroke to type 'x'
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, AppProps { actors });
            })
            .unwrap();

        let buffer = terminal.backend().buffer().clone();
        let buffer_content = format!("{:?}", buffer);

        // The filter should now contain "testx"
        assert!(buffer_content.contains("testx"));
    }

    #[test]
    fn test_cursor_in_middle_type() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with some text in filter
        let mut app = App::new("test_execution_id".to_string());

        // Add some text to the filter
        for c in "test".chars() {
            app.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Set cursor position to middle (after 't')
        app.set_cursor_position_unit_tests_only(1);

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Send a keystroke to type 'x'
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, AppProps { actors });
            })
            .unwrap();

        let buffer = terminal.backend().buffer().clone();
        let buffer_content = format!("{:?}", buffer);

        // The filter should now contain "txest"
        assert!(buffer_content.contains("txest"));
    }

    #[test]
    fn test_cursor_at_zero_ctrl_e() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with some text in filter
        let mut app = App::new("test_execution_id".to_string());

        // Add some text to the filter
        for c in "test".chars() {
            app.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Set cursor position to 0
        app.set_cursor_position_unit_tests_only(0);

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Send Ctrl+E keystroke
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('e'),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Now type a character to verify the cursor is at the end
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        let buffer = terminal.backend().buffer().clone();
        let buffer_content = format!("{:?}", buffer);

        // The filter should now contain "testx" (not "xtest")
        // This verifies that Ctrl+E moved the cursor to the end
        assert!(buffer_content.contains("testx"));
    }

    #[test]
    fn test_cursor_at_end_ctrl_a() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with some text in filter
        let mut app = App::new("test_execution_id".to_string());

        // Add some text to the filter
        for c in "test".chars() {
            app.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Set cursor position to end
        app.set_cursor_position_unit_tests_only(4); // "test" has 4 characters

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Send Ctrl+A keystroke
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Now type a character to verify the cursor is at the beginning
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, AppProps { actors });
            })
            .unwrap();

        let buffer = terminal.backend().buffer().clone();
        let buffer_content = format!("{:?}", buffer);

        // The filter should now contain "xtest" (not "testx")
        // This verifies that Ctrl+A moved the cursor to the beginning
        assert!(buffer_content.contains("xtest"));
    }

    #[test]
    fn test_arrow_keys_cursor_movement() {
        // Create a test backend with a fixed size
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an app with some text in filter
        let mut app = App::new("test_execution_id".to_string());

        // Add some text to the filter
        let test_text = "hello world";
        for c in test_text.chars() {
            app.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Set cursor position to middle of the text
        let middle_position = 5; // After "hello"
        app.set_cursor_position_unit_tests_only(middle_position);

        // Create actors vector for props
        let actors = vec![ActorInfo {
            actor_id: "test.actor1".to_string(),
            message_count: 10,
        }];

        // Render the app to capture initial state
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Send left arrow key to move cursor left
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Left,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Type a character to verify cursor position
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('X'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(
                    f,
                    area,
                    AppProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        let buffer = terminal.backend().buffer().clone();
        let buffer_content = format!("{:?}", buffer);

        // The filter should now contain "hellXo world" (X inserted before 'o')
        assert!(buffer_content.contains("hellXo world"));

        // Now move cursor right twice
        for _ in 0..2 {
            app.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Type another character to verify new cursor position
        app.handle(&crossterm::event::KeyEvent {
            code: KeyCode::Char('Y'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the app again to see the effect
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 20);
                app.render(f, area, AppProps { actors });
            })
            .unwrap();

        let buffer = terminal.backend().buffer().clone();
        let buffer_content = format!("{:?}", buffer);

        // The filter should now contain "hellXo Yworld" (Y inserted after space)
        assert!(buffer_content.contains("hellXo Yworld"));
    }

    #[test]
    fn test_app_table_truncates_actors() {
        // Create a test backend with a larger size to accommodate 100 actors
        let backend = TestBackend::new(120, 26);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an empty AppTable
        let mut app_table = AppTable::new();

        // Create props with 100 actors
        let mut actors = Vec::with_capacity(100);
        for i in 1..50 {
            actors.push(ActorInfo {
                actor_id: format!("test.actor{}", i),
                message_count: i,
            });
        }

        let props = AppTableProps { actors };

        // Render the table
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 120, 150);
                app_table.render(f, area, props);
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let buffer = terminal.backend().buffer().clone();

        // Convert buffer to string for easier assertions
        let buffer_content = format!("{:?}", buffer);
        eprintln!("Buffer content: {}", buffer_content);

        for i in 1..=25 {
            assert!(buffer_content.contains(&format!("test.actor{}", i)));
        }
        for i in 26..=50 {
            assert!(!buffer_content.contains(&format!("test.actor{}", i)));
        }
        let actor_count = buffer_content.matches("test.actor").count();
        assert!(actor_count == 25);
    }

    #[test]
    fn test_app_table_scrolling() {
        // Create a test backend with a small size to test scrolling
        // Height of 10 means we can only see about 9 actors (1 row for header)
        let backend = TestBackend::new(80, 10);
        let mut terminal = Terminal::new(backend).unwrap();

        // Create an AppTable
        let mut app_table = AppTable::new();

        // Create props with 20 actors
        let mut actors = Vec::with_capacity(20);
        for i in 1..=30 {
            actors.push(ActorInfo {
                actor_id: format!("test.actor{}", i),
                message_count: i * 10,
            });
        }

        // Render the table initially (should show actors 1-9)
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 10);
                app_table.render(
                    f,
                    area,
                    AppTableProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Get the buffer to check what was rendered
        let initial_buffer = terminal.backend().buffer().clone();
        let initial_content = format!("{:?}", initial_buffer);
        eprintln!("{}", initial_content);

        // Verify that the first few actors are visible
        for i in 1..=9 {
            assert!(initial_content.contains(&format!("test.actor{}", i)));
        }
        // Verify that later actors are not visible
        assert!(!initial_content.contains("test.actor10"));

        // Scroll down 5 rows using Down key
        for _ in 0..4 {
            app_table.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }

        // Render the table again
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 10);
                app_table.render(
                    f,
                    area,
                    AppTableProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Get the buffer to check what was rendered after scrolling
        let scrolled_buffer = terminal.backend().buffer().clone();
        let scrolled_content = format!("{:?}", scrolled_buffer);
        eprintln!("{}", scrolled_content);

        // Verify that later actors are now visible
        for i in 5..=13 {
            assert!(scrolled_content.contains(&format!("test.actor{}", i)));
        }

        // Verify that the first few actors are no longer visible
        assert!(!scrolled_content.contains("test.actor4 "));
        // Verify that the next actors are no longer visible
        assert!(!scrolled_content.contains("test.actor14 "));

        // Scroll down 100 times using Down key
        for _ in 0..100 {
            app_table.handle(&crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::empty(),
                kind: KeyEventKind::Press,
                state: crossterm::event::KeyEventState::empty(),
            });
        }
        // Render the table again
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 10);
                app_table.render(
                    f,
                    area,
                    AppTableProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Get the buffer to check what was rendered after scrolling
        let scrolled_buffer = terminal.backend().buffer().clone();
        let scrolled_content = format!("{:?}", scrolled_buffer);
        eprintln!("{}", scrolled_content);
        for i in 22..=30 {
            assert!(scrolled_content.contains(&format!("test.actor{}", i)));
        }

        // Scroll back up using PageUp
        app_table.handle(&crossterm::event::KeyEvent {
            code: KeyCode::PageUp,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::empty(),
        });

        // Render the table again
        terminal
            .draw(|f| {
                let area = Rect::new(0, 0, 80, 10);
                app_table.render(
                    f,
                    area,
                    AppTableProps {
                        actors: actors.clone(),
                    },
                );
            })
            .unwrap();

        // Get the buffer to check what was rendered after scrolling back up
        let scrolled_up_buffer = terminal.backend().buffer().clone();
        let scrolled_up_content = format!("{:?}", scrolled_up_buffer);
        eprintln!("{}", scrolled_up_content);

        // Verify that even after scrolling way past the end we immediately no longer see the last actor
        assert!(!scrolled_content.contains("test.actor49 "));
    }

    #[tokio::test]
    async fn test_exits() {
        let mut app = App::new("test id".to_string());
        app.handle(&KeyCode::Char(':').into());
        app.handle(&KeyCode::Char('q').into());
        app.handle(&KeyCode::Enter.into());
        assert!(app.exit);
    }
}
