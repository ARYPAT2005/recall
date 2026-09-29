//! Splits text into lowercase words.

/// "Machine Learning, fast!" -> ["machine", "learning", "fast"]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for_each_token(text, &mut String::new(), |t| out.push(t.to_owned()));
    out
}

/// Like `tokenize`, but calls `f` on each word instead of allocating a String for it.
pub fn for_each_token(text: &str, buf: &mut String, mut f: impl FnMut(&str)) {
    let mut emit = |word: &str, upper: bool, non_ascii: bool| {
        if non_ascii {
            f(&word.to_lowercase());
        } else if upper {
            buf.clear();
            buf.push_str(word);
            buf.make_ascii_lowercase();
            f(buf);
        } else {
            f(word);
        }
    };

    let mut start = None;
    let (mut upper, mut non_ascii) = (false, false);
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            start.get_or_insert(i);
            upper |= c.is_ascii_uppercase();
            non_ascii |= !c.is_ascii();
        } else if let Some(s) = start.take() {
            emit(&text[s..i], upper, non_ascii);
            (upper, non_ascii) = (false, false);
        }
    }
    if let Some(s) = start {
        emit(&text[s..], upper, non_ascii);
    }
}
