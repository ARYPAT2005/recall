//! Text to lowercase terms, shared by indexing and queries so both see the
//! same words.

/// "Machine Learning, fast!" -> ["machine", "learning", "fast"]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for_each_token(text, &mut String::new(), |t| out.push(t.to_owned()));
    out
}

/// The allocation-free tokenizer behind `tokenize`: calls `f` with each
/// lowercase token. A token is a maximal run of alphanumeric chars.
///
/// Already-lowercase ASCII words (almost all of them) are passed as a slice of
/// `text` with no copy. Words with uppercase ASCII are lowercased into `buf`,
/// which is reused, so indexing a document does no per-token allocation. Only
/// non-ASCII words fall back to `str::to_lowercase`, which handles cases like
/// Greek final sigma that char-by-char lowercasing gets wrong.
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

    let mut start = None; // byte offset where the current token began
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
