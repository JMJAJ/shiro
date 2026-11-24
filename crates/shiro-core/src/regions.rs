//! Cheap enwik region classifier, used only for reporting where bits go.
//! It never influences prediction, so it does not need to be exact.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    Header = 0,   // XML outside <text> (page/revision headers, siteinfo)
    Tag = 1,      // inside <...> or &entities; within text
    Link = 2,     // inside [[ ]]
    Template = 3, // inside {{ }} or tables {| |}
    Text = 4,     // article prose
}

pub const N_REGIONS: usize = 5;
pub const NAMES: [&str; N_REGIONS] = ["header", "tag", "link", "template", "text"];

pub struct Classifier {
    last: u64,
    in_text: bool,
    in_tag: bool,
    link: u32,
    tmpl: u32,
}

impl Default for Classifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier {
    pub fn new() -> Self {
        Classifier { last: 0, in_text: false, in_tag: false, link: 0, tmpl: 0 }
    }

    pub fn region(&self) -> Region {
        if !self.in_text {
            Region::Header
        } else if self.in_tag {
            Region::Tag
        } else if self.tmpl > 0 {
            Region::Template
        } else if self.link > 0 {
            Region::Link
        } else {
            Region::Text
        }
    }

    fn ends_with(&self, s: &[u8]) -> bool {
        let n = s.len();
        let mut v = 0u64;
        for &b in s {
            v = (v << 8) | b as u64;
        }
        let mask = if n >= 8 { u64::MAX } else { (1u64 << (8 * n)) - 1 };
        self.last & mask == v
    }

    pub fn push(&mut self, b: u8) {
        self.last = (self.last << 8) | b as u64;
        if !self.in_text {
            // enter text after the '>' that closes "<text ...>"
            if self.ends_with(b"<text") {
                self.in_tag = true;
            }
            if self.in_tag && b == b'>' {
                self.in_tag = false;
                self.in_text = true;
                self.link = 0;
                self.tmpl = 0;
            }
            return;
        }
        if self.ends_with(b"</text") {
            self.in_text = false;
            self.in_tag = false;
            return;
        }
        match b {
            b'&' => self.in_tag = true,
            b';' if self.in_tag => self.in_tag = false,
            b' ' | b'\n' if self.in_tag => self.in_tag = false,
            _ => {}
        }
        if self.ends_with(b"[[") {
            self.link += 1;
        } else if self.ends_with(b"]]") {
            self.link = self.link.saturating_sub(1);
        } else if self.ends_with(b"{{") || self.ends_with(b"{|") {
            self.tmpl += 1;
        } else if self.ends_with(b"}}") || self.ends_with(b"|}") {
            self.tmpl = self.tmpl.saturating_sub(1);
        }
        if self.link > 8 {
            self.link = 0;
        }
        if self.tmpl > 8 {
            self.tmpl = 0;
        }
    }
}
