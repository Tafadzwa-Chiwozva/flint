//! Streaming detokenizer filter.
//!
//! OpenAI clients treat each SSE chunk as an append. A stop string must not
//! leak into those chunks, so we hold back `max_stop_len - 1` characters until
//! either the stop sequence is complete (and dropped) or generation ends.

/// Holds back a suffix that might still grow into a stop string.
#[derive(Debug)]
pub(crate) struct StreamFilter {
    stops: Vec<String>,
    hold: String,
}

impl StreamFilter {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|stop| !stop.is_empty()).collect(),
            hold: String::new(),
        }
    }

    /// Append newly decoded text.
    ///
    /// Returns the text that is safe to emit, and whether a stop string was
    /// found. The stop string itself is not part of the returned text.
    pub(crate) fn push_delta(&mut self, delta: &str) -> (String, bool) {
        self.hold.push_str(delta);
        if let Some(index) = self.find_stop() {
            let visible = self.hold[..index].to_string();
            self.hold.clear();
            return (visible, true);
        }
        let keep = self.max_stop_chars().saturating_sub(1);
        let chars = self.hold.chars().count();
        if chars <= keep {
            return (String::new(), false);
        }
        let emit_chars = chars - keep;
        let split = self
            .hold
            .char_indices()
            .nth(emit_chars)
            .map(|(index, _)| index)
            .unwrap_or(self.hold.len());
        let visible = self.hold[..split].to_string();
        self.hold.drain(..split);
        (visible, false)
    }

    /// Release the held suffix. Called when generation ends for a reason other
    /// than a stop string, so the last characters are not stuck in the filter.
    pub(crate) fn flush(&mut self) -> String {
        std::mem::take(&mut self.hold)
    }

    fn max_stop_chars(&self) -> usize {
        self.stops
            .iter()
            .map(|stop| stop.chars().count())
            .max()
            .unwrap_or(0)
    }

    fn find_stop(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.hold.find(stop))
            .min()
    }
}

/// Characters added by a growing decode.
///
/// Hugging Face detokenization of a prefix of ids is usually a prefix of the
/// string, except when a trailing partial codepoint is withheld and then
/// completed. If `next` does not extend `previous`, only the suffix past the
/// common prefix is returned.
pub(crate) fn text_delta(previous: &str, next: &str) -> String {
    if let Some(rest) = next.strip_prefix(previous) {
        return rest.to_string();
    }
    let common = previous
        .char_indices()
        .zip(next.chars())
        .take_while(|((_, ch_prev), ch_next)| ch_prev == ch_next)
        .map(|((index, ch), _)| index + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if common >= next.len() {
        String::new()
    } else {
        next[common..].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_back_a_partial_stop_then_drops_it() {
        let mut filter = StreamFilter::new(vec!["AA".into()]);
        assert_eq!(filter.push_delta("A"), (String::new(), false));
        assert_eq!(filter.push_delta("A"), (String::new(), true));
        assert_eq!(filter.flush(), "");
    }

    #[test]
    fn emits_text_that_cannot_grow_into_a_stop() {
        let mut filter = StreamFilter::new(vec!["STOP".into()]);
        let (visible, hit) = filter.push_delta("Hello");
        assert!(!hit);
        assert_eq!(visible, "He");
        assert_eq!(filter.flush(), "llo");
    }

    #[test]
    fn stop_in_the_middle_keeps_the_prefix() {
        let mut filter = StreamFilter::new(vec!["\n".into()]);
        let (visible, hit) = filter.push_delta("ab\ncd");
        assert!(hit);
        assert_eq!(visible, "ab");
    }

    #[test]
    fn empty_stop_list_emits_everything() {
        let mut filter = StreamFilter::new(vec![]);
        assert_eq!(filter.push_delta("Hi"), ("Hi".into(), false));
    }

    #[test]
    fn text_delta_is_the_appended_suffix() {
        assert_eq!(text_delta("Hel", "Hello"), "lo");
        assert_eq!(text_delta("Hello", "Hello"), "");
        assert_eq!(text_delta("", "A"), "A");
    }
}
