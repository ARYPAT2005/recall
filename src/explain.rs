//! `explain`: what the engine did for one query, and why.
//!
//! A Plan is filled in by the same code path that answers real queries
//! (`Index::matching` with tracing on), so it can't drift from what search
//! actually does. With tracing off that path takes no timestamps at all.

use crate::{Hit, Strategy, GALLOP_RATIO};
use std::fmt;
use std::time::Duration;

/// One query term as the planner saw it.
#[derive(Debug, Clone)]
pub struct PlanTerm {
    pub term: String,
    /// Document frequency. 0 means the term isn't in the index.
    pub df: usize,
    pub idf: f32,
}

/// One intersection: the running candidates against the next term's list.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub term: String,
    /// Candidates going in (always the shorter side).
    pub candidates: usize,
    /// Length of the term's posting list.
    pub df: usize,
    /// What Adaptive resolved to: Merge or Gallop.
    pub algorithm: Strategy,
    /// Candidates left afterwards.
    pub matched: usize,
    pub time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub query: String,
    pub num_docs: usize,
    /// Distinct query terms in execution order, rarest first.
    pub terms: Vec<PlanTerm>,
    /// Tokenizing, deduping, looking up and ordering the terms.
    pub lookup_time: Duration,
    /// Intersections, in order. Fewer than terms - 1 if candidates ran out.
    pub steps: Vec<PlanStep>,
    pub matched: usize,
    pub k: usize,
    pub hits: Vec<Hit>,
    /// Scoring every match with BM25 and selecting the top k.
    pub rank_time: Duration,
}

impl Plan {
    pub fn total_time(&self) -> Duration {
        self.lookup_time + self.steps.iter().map(|s| s.time).sum::<Duration>() + self.rank_time
    }
}

/// 1234567 -> "1,234,567"
fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn micros(d: Duration) -> String {
    format!("{:.2} µs", d.as_secs_f64() * 1e6)
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QUERY PLAN  {:?}  over {} docs", self.query, grouped(self.num_docs))?;

        if self.terms.is_empty() {
            return writeln!(f, "\n  No terms in the query.");
        }

        let width = self.terms.iter().map(|t| t.term.len()).max().unwrap_or(0) + 2;
        writeln!(f, "\nTerms, rarest first (that's the execution order):")?;
        for t in &self.terms {
            let quoted = format!("{:?}", t.term);
            if t.df == 0 {
                writeln!(f, "  {quoted:<width$}  not in the index")?;
            } else {
                writeln!(f, "  {quoted:<width$}  df {:>9}   idf {:>5.2}", grouped(t.df), t.idf)?;
            }
        }

        writeln!(f, "\nSteps:")?;
        let first = &self.terms[0];
        writeln!(
            f,
            "  1. tokenize, look up and order {} term(s), start from {:?}: {} candidates   {}",
            self.terms.len(),
            first.term,
            grouped(first.df),
            micros(self.lookup_time)
        )?;
        let mut n = 1;
        for s in &self.steps {
            n += 1;
            let ratio = s.df as f64 / s.candidates.max(1) as f64;
            let why = match s.algorithm {
                Strategy::Gallop => format!("{ratio:.1}x >= {GALLOP_RATIO}x: gallop"),
                _ => format!("{ratio:.1}x < {GALLOP_RATIO}x: merge"),
            };
            writeln!(
                f,
                "  {n}. AND {:?}: {} vs {}, {why} -> {} left   {}",
                s.term,
                grouped(s.candidates),
                grouped(s.df),
                grouped(s.matched),
                micros(s.time)
            )?;
        }

        // Candidates ran out before every term was used: the rest were
        // never touched. Say so, since that's the payoff of rarest-first.
        let skipped = &self.terms[1 + self.steps.len()..];
        if !skipped.is_empty() {
            let names: Vec<String> = skipped.iter().map(|t| format!("{:?}", t.term)).collect();
            writeln!(f, "     no candidates left, so {} never read", names.join(", "))?;
        }

        if self.matched > 0 {
            n += 1;
            let select = if self.k >= self.matched {
                "sort them all".to_string()
            } else {
                format!("keep the best {} in a heap", self.k)
            };
            writeln!(
                f,
                "  {n}. BM25-score {} match(es), {select}   {}",
                grouped(self.matched),
                micros(self.rank_time)
            )?;
        }

        writeln!(
            f,
            "\nTotal {}: {} matched, returning {}. (One run; steps under a microsecond are near the timer's resolution.)",
            micros(self.total_time()),
            grouped(self.matched),
            self.hits.len()
        )
    }
}
