# TTS Text Preprocessing Specification

## Overview

This document specifies a two-layer text preprocessing system for Rokoko TTS
when used as a Home Assistant voice backend. The goal is to transform arbitrary
text into speech-ready prose that the Kokoro TTS model can pronounce cleanly.

**Architecture:**

```
User query
    |
    v
[LLM pre-speech step]   <-- shapes content: brevity, style, no markup
    |
    v
[Deterministic rules]   <-- mechanical cleanup: symbols, numbers, formatting
    |
    v
[Existing normalize.h]  <-- money, dates, standalone numbers, unicode->ascii
    |
    v
[G2P -> TTS -> audio]
```

The LLM handles semantic decisions (what to say, how to phrase it). The
deterministic rules act as a safety net catching anything the LLM leaks.
`normalize.h` already handles several categories and must not be duplicated.

---

## What Already Exists in `normalize.h`

The current `text_norm::preprocess_text()` pipeline runs 5 stages:

| Stage | What it does | Examples |
|-------|-------------|----------|
| 1. Money | Currency symbol + amount -> words | `$12.50` -> `twelve dollars and fifty cents` |
| 2. Dates | US, ISO, textual dates -> words | `1/15/2024` -> `January fifteenth, twenty twenty four` |
| 3. Numbers | Standalone integers -> words | `1234` -> `one thousand two hundred thirty four` |
| 4-5. Unicode | Latin diacritics -> ASCII, strip non-printable | `cafe` -> `cafe`, em-dash -> `--` |

**Known gaps in normalize.h** (things it does NOT handle):

- Decimals (`3.14`)
- Negative numbers (`-5`)
- Ordinal suffixes (`1st`, `2nd`, `3rd`)
- Percentages (`50%`)
- Times (`3:30 PM`)
- Fractions (`3/4`)
- Units/measurements (`5kg`, `60mph`)
- Phone numbers
- Acronyms spelled letter-by-letter
- Version numbers (`v2.1.3`)
- Ranges (`3-5`)

These gaps must be filled by the new deterministic rules layer, which runs
**before** `normalize.h` in the pipeline.

---

## Layer 1: LLM System Prompt Rules

These rules are enforced via the system prompt of whatever LLM generates the
response that will be spoken. They require semantic understanding and cannot be
reliably done with regex.

### 1.1 Response Length

Limit responses to **4 sentences maximum**. Prefer 2-3 sentences when possible.
Long spoken responses are hard to follow and feel robotic.

### 1.2 Paragraph-Only Output

Output must be a single plain paragraph. No structured formatting of any kind:

- No bullet points or numbered lists
- No tables
- No markdown headers (`#`, `##`, etc.)
- No horizontal rules (`---`)
- No block quotes (`>`)

If the information naturally wants to be a list, rephrase it as prose:
- BAD: `"Features: - Fast - Lightweight - Easy to use"`
- GOOD: `"It's fast, lightweight, and easy to use."`

### 1.3 No Markup or Formatting

No markdown syntax anywhere in the response:

- No bold (`**text**` or `__text__`)
- No italic (`*text*` or `_text_`)
- No inline code (`` `code` ``)
- No fenced code blocks (`` ``` ``)
- No strikethrough (`~~text~~`)
- No links (`[text](url)`)
- No images (`![alt](url)`)

### 1.4 No Emoji

Never include emoji characters. Not even common ones like thumbs up or checkmarks.

### 1.5 No Math Notation

Express all math and formulas in natural spoken language:

- BAD: `"2 + 2 = 4"`
- GOOD: `"Two plus two equals four."`
- BAD: `"x^2 + y^2 = z^2"`
- GOOD: `"X squared plus Y squared equals Z squared."`
- BAD: `"The ratio is 3:1"`
- GOOD: `"The ratio is three to one."`

### 1.6 No Code

Do not include function names, variable names, file paths, stack traces, code
blocks, or any programming syntax in the spoken response. Describe them
conversationally instead:

- BAD: `"Call getUserName() to fetch the username"`
- GOOD: `"Use the get user name function to fetch the username."`
- BAD: `"Edit /etc/nginx/nginx.conf"`
- GOOD: `"Edit the nginx configuration file."`

### 1.7 No URLs or Email Addresses

Do not include raw URLs or email addresses. Describe where to find things:

- BAD: `"Go to https://example.com/settings"`
- GOOD: `"Go to the settings page on example dot com."`

### 1.8 Expand Abbreviations

Use full words, not abbreviations:

- `"e.g."` -> `"for example"`
- `"i.e."` -> `"that is"`
- `"vs."` -> `"versus"`
- `"etc."` -> `"and so on"` or `"et cetera"`
- `"approx."` -> `"approximately"`
- `"dept."` -> `"department"`
- `"govt."` -> `"government"`

### 1.9 Avoid Parenthetical Asides

Parenthetical expressions break speech flow. Rephrase as separate sentences
or integrate into the main clause:

- BAD: `"The server (which runs on port 8080) is healthy."`
- GOOD: `"The server is healthy. It runs on port 8080."`

### 1.10 Avoid Ambiguous Homographs

Where possible, rephrase to avoid words with multiple pronunciations:

- `"read"` (present vs. past) -> use `"reading"` or `"already read it"`
- `"live"` (verb vs. adjective) -> context should make it clear
- `"lead"` (metal vs. verb) -> use `"leading"` or `"the metal lead"`
- `"bass"` -> `"bass guitar"` or `"the fish called bass"`

### 1.11 Numbers

Use words for 1-12 in conversational contexts. Larger numbers can be digits
(the TTS normalizer converts them). Avoid bare numbers at the start of
sentences.

---

## Layer 2: Deterministic Rules

These are regex/string replacement rules applied to the LLM's output **before**
it reaches `normalize.h`. They are organized by processing order. Each stage
is independent of subsequent stages but may depend on prior stages having run.

Implementation note: These should be a new function (e.g.,
`speech_prep::prepare_for_tts()`) that runs before `text_norm::preprocess_text()`.

---

### Stage 1: Strip Surviving Markup

Even with LLM instructions, markdown/HTML occasionally leaks through. These
rules act as a safety net.

**1.1 — Strip markdown bold/italic**
```
Pattern:  \*{1,3}(.+?)\*{1,3}
Replace:  \1
Example:  "This is **important**" -> "This is important"
```

**1.2 — Strip markdown underline bold/italic**
```
Pattern:  _{1,3}(.+?)_{1,3}
Replace:  \1
Note:     Only match when underscores wrap a word/phrase, not snake_case.
          Use negative lookbehind/lookahead for word chars:
          (?<!\w)_{1,3}(.+?)_{1,3}(?!\w)
Example:  "__bold__" -> "bold"
          "my_var_name" -> unchanged
```

**1.3 — Strip inline code backticks**
```
Pattern:  `([^`]+)`
Replace:  \1
Example:  "Run `make install`" -> "Run make install"
```

**1.4 — Strip fenced code blocks**
```
Pattern:  ```[\s\S]*?```     (multiline)
Replace:  (empty string)
Note:     Remove the entire block including content. Code is not speakable.
Example:  "Here's the code:\n```python\nprint('hi')\n```\nThat's it."
       -> "Here's the code:\n\nThat's it."
```

**1.5 — Strip markdown links, keep link text**
```
Pattern:  \[([^\]]+)\]\([^)]+\)
Replace:  \1
Example:  "See [the docs](https://example.com)" -> "See the docs"
```

**1.6 — Strip markdown images**
```
Pattern:  !\[[^\]]*\]\([^)]+\)
Replace:  (empty string)
Example:  "Look: ![screenshot](img.png)" -> "Look: "
```

**1.7 — Strip markdown headers**
```
Pattern:  ^#{1,6}\s+      (per line)
Replace:  (empty string)
Example:  "## Settings" -> "Settings"
```

**1.8 — Strip bullet/list prefixes**
```
Pattern:  ^\s*[-*+\u2022]\s+   (per line, unicode bullet included)
Replace:  (empty string)
Example:  "- First item" -> "First item"
```

**1.9 — Strip numbered list prefixes**
```
Pattern:  ^\s*\d+[.)]\s+       (per line)
Replace:  (empty string)
Example:  "1. First item" -> "First item"
```

**1.10 — Strip horizontal rules**
```
Pattern:  ^[-*_]{3,}\s*$        (per line)
Replace:  (empty string)
```

**1.11 — Strip HTML tags**
```
Pattern:  <[^>]+>
Replace:  (empty string)
Example:  "<b>bold</b>" -> "bold"
```

**1.12 — Expand HTML entities**
```
&amp;   -> "and"
&lt;    -> "less than"
&gt;    -> "greater than"
&nbsp;  -> " "
&quot;  -> (double quote char)
&#39;   -> (apostrophe)
```

**1.13 — Strip block quotes**
```
Pattern:  ^>\s?               (per line)
Replace:  (empty string)
Example:  "> This is quoted" -> "This is quoted"
```

---

### Stage 2: Unicode Normalization (supplements to normalize.h)

`normalize.h` handles Latin diacritics and basic punctuation. These rules cover
additional Unicode characters it does not handle.

**2.1 — Smart quotes to ASCII**
```
\u201c (left double quote)   -> "
\u201d (right double quote)  -> "
\u2018 (left single quote)   -> '
\u2019 (right single quote)  -> '
\u201e (double low-9 quote)  -> "
\u201a (single low-9 quote)  -> '
```

**2.2 — Dashes to speech-friendly punctuation**
```
\u2014 (em dash)  -> ", "  (comma + space — creates a natural pause)
\u2013 (en dash)  -> ", "  (same treatment; context-dependent "to" handled in Stage 5)
```
Note: `normalize.h` converts em-dash to `--` and en-dash to `-`. The new rules
should run BEFORE `normalize.h` to intercept these first. Alternatively, modify
the unicode table in `normalize.h` to produce `, ` instead of `--` for em-dash.

**2.3 — Ellipsis**
```
\u2026 (horizontal ellipsis) -> "..."
```
Already handled by `normalize.h` (maps to `...`). No change needed.

**2.4 — Invisible/zero-width characters**
```
\u200b (zero-width space)    -> (strip)
\u200c (zero-width non-joiner) -> (strip)
\u200d (zero-width joiner)   -> (strip)
\ufeff (BOM / zero-width no-break space) -> (strip)
\u00ad (soft hyphen)         -> (strip)
\u200e (left-to-right mark)  -> (strip)
\u200f (right-to-left mark)  -> (strip)
\u2060 (word joiner)         -> (strip)
```

**2.5 — Non-breaking space**
```
\u00a0 -> regular space (0x20)
```

**2.6 — Miscellaneous Unicode symbols**
```
\u00b0 (degree sign)   -> " degrees "
\u00a9 (copyright)     -> "copyright "
\u00ae (registered)    -> "registered "
\u2122 (trademark)     -> "trademark "   (normalize.h maps to "TM")
\u00b1 (plus-minus)    -> "plus or minus "
\u00d7 (multiplication)-> " times "
\u00f7 (division)      -> " divided by "
\u2022 (bullet)        -> (strip, handled in Stage 1 list removal)
```

---

### Stage 3: Arrow and Sequence Symbols

Arrows indicate sequences or transitions. Replace with spoken equivalents.

**3.1 — Unicode arrows**
```
\u2192 (rightward arrow ->)  -> " then "
\u2190 (leftward arrow <-)   -> " from "
\u2194 (left-right arrow)    -> " to and from "
\u21d2 (rightward double =>)  -> " then "
\u21d0 (leftward double <=)  -> " from "
\u2191 (upward arrow)        -> "up"
\u2193 (downward arrow)      -> "down"
\u2197 (upper-right arrow)   -> "up"
\u2198 (lower-right arrow)   -> "down"
```

**3.2 — ASCII arrows**
```
Pattern:  \s*-{1,2}>\s*
Replace:  " then "
Example:  "Move -> Jump -> Attack" -> "Move then Jump then Attack"

Pattern:  \s*=>\s*
Replace:  " then "
Example:  "A => B => C" -> "A then B then C"

Pattern:  \s*<-{1,2}\s*
Replace:  " from "
Example:  "result <- input" -> "result from input"
```

---

### Stage 4: Symbol Verbalization

Replace symbols with their spoken equivalents. Order matters: some rules depend
on context established by earlier rules.

**4.1 — Ampersand**
```
Pattern:  \s*&\s*
Replace:  " and "
Example:  "R&D" -> "R and D"
          "Tom & Jerry" -> "Tom and Jerry"
```

**4.2 — At sign (non-email)**
```
Pattern:  @(?!\S+\.\S+)    (negative lookahead: not email-shaped)
Replace:  " at "
Example:  "@home" -> " at home"

For emails (if they leak through LLM):
Pattern:  (\S+)@(\S+)\.(\S+)
Replace:  "\1 at \2 dot \3"
Example:  "user@example.com" -> "user at example dot com"
```

**4.3 — Hash/pound sign**
```
Before digits:  #(\d) -> "number \1"
Before words:   #(\w) -> (strip the #)
Standalone:     # -> (strip)
Example:  "#1" -> "number 1"
          "#OpenSource" -> "OpenSource"
```

**4.4 — Math operators (in mathematical context)**

These should be applied only when surrounded by spaces or between digits,
to avoid false positives in normal prose.

```
Pattern:  (\d)\s*\+\s*(\d)
Replace:  \1 plus \2
Example:  "2 + 2" -> "2 plus 2"

Pattern:  (\d)\s*-\s*(\d)
Replace:  \1 minus \2
Caution:  Conflicts with ranges (Stage 5) and hyphens. Only apply in
          explicit math contexts (both sides are digits with spaces).
          Use: (\d)\s+-\s+(\d) (require spaces) for minus.

Pattern:  (\d)\s*[x\u00d7]\s*(\d)
Replace:  \1 times \2
Example:  "3 x 4" -> "3 times 4"

Pattern:  (\d)\s*[/\u00f7]\s*(\d)    (but NOT date patterns)
Replace:  \1 divided by \2
Caution:  Only when clearly math, not fractions handled in Stage 5.

Pattern:  \s*=\s*
Replace:  " equals "
Example:  "2 + 2 = 4" -> "2 plus 2 equals 4"
Caution:  Don't apply inside URLs or code that leaked through. Safe to
          apply when both sides have content.

Pattern:  <(?!=)    (less-than, not <=)
Replace:  " less than "

Pattern:  >(?!=)    (greater-than, not >=)
Replace:  " greater than "
Caution:  Run AFTER HTML tag stripping (Stage 1.11).

Pattern:  <=
Replace:  " less than or equal to "

Pattern:  >=
Replace:  " greater than or equal to "

Pattern:  !=
Replace:  " not equal to "
```

**4.5 — Percent**
```
Pattern:  (\d)\s*%
Replace:  \1 percent
Example:  "50%" -> "50 percent"
Note:     normalize.h will then expand "50" to "fifty".
```

**4.6 — Caret (exponentiation)**
```
Pattern:  (\w)\^2\b
Replace:  \1 squared
Example:  "x^2" -> "x squared"

Pattern:  (\w)\^3\b
Replace:  \1 cubed
Example:  "x^3" -> "x cubed"

Pattern:  (\w)\^(\d+)
Replace:  \1 to the power of \2
Example:  "10^6" -> "10 to the power of 6"
```

**4.7 — Pipe**
```
Pattern:  \s*\|\s*
Replace:  " or "   (in most conversational contexts)
Example:  "yes | no" -> "yes or no"
```

**4.8 — Backslash**
```
Pattern:  \\
Replace:  " "
Example:  "path\\to\\file" -> "path to file"
```

**4.9 — Underscore (word separator)**
```
Pattern:  _
Replace:  " "
Context:  snake_case identifiers and similar.
Example:  "queue_entry" -> "queue entry"
          "my_variable_name" -> "my variable name"
```

**4.10 — Brackets and parentheses (strip containers, keep content)**
```
Pattern:  [(){}\[\]]
Replace:  (empty string)
Example:  "queue_entry()" -> "queue_entry"  (underscore rule then: "queue entry")
          "array[0]" -> "array0"  (number adjacency will keep "0" as digit)
          "{value}" -> "value"
```

**4.11 — Tilde**
```
Pattern:  ~(\d)
Replace:  "approximately \1"
Example:  "~100" -> "approximately 100"

Pattern:  ~(?!\d)
Replace:  (strip)
```

**4.12 — Asterisk (not in markdown context, already stripped)**
```
Pattern:  \*
Replace:  (strip)
Note:     By this stage, markdown bold markers are already removed. Remaining
          asterisks are footnote markers or decoration. Strip them.
```

**4.13 — Forward slash (context-dependent)**
```
In units (after stage 5 unit expansion): already handled.
In alternatives:
Pattern:  (\w)\s*/\s*(\w)
Replace:  \1 or \2
Example:  "his/her" -> "his or her"
          "and/or" -> "and or"
Caution:  Don't apply to fractions (handled in Stage 5) or dates (handled
          by normalize.h). Apply AFTER those stages, or use negative
          lookahead for digit/digit patterns.
```

**4.14 — Dollar sign (without amount, leaked through)**
```
If $ not followed by digit (already handled by normalize.h for $amounts):
Pattern:  \$(?!\d)
Replace:  "dollar "
```

---

### Stage 5: Quantities, Measurements, and Number Edge Cases

These rules handle numeric patterns that `normalize.h` does not cover. They
must run BEFORE `normalize.h` so that any digits they produce are then
converted to words by the existing number expansion.

**5.1 — Ordinal suffixes**
```
Pattern:  (\d+)(st|nd|rd|th)\b
Action:   Convert to ordinal words directly.
          1st -> "first", 2nd -> "second", 3rd -> "third", 4th -> "fourth"
          11th -> "eleventh", 12th -> "twelfth", 13th -> "thirteenth"
          21st -> "twenty first", 22nd -> "twenty second"
          ...and so on up to reasonable limits.
Fallback: For large ordinals (beyond a lookup table), expand the number
          to words and append "th" (e.g., 100th -> "one hundredth").
Example:  "the 3rd floor" -> "the third floor"
```

**5.2 — Decimal numbers**
```
Pattern:  (\d+)\.(\d+)(?!\S*\d*[/-]\d)    (not a date or version)
Replace:  \1 point \2-expanded-digit-by-digit
Example:  "3.14" -> "3 point 1 4"
          normalize.h then: "three point one four"
Strategy: Split on decimal point. Left side stays as number (normalize.h
          handles it). Right side: spell each digit individually separated
          by spaces so normalize.h expands each one.
Edge:     "0.5" -> "zero point five" (normalize.h handles "0" -> "zero")
```

**5.3 — Negative numbers**
```
Pattern:  (?<=\s|^)-(\d)
Replace:  "negative \1"
Example:  "-5" -> "negative 5" -> (normalize.h) -> "negative five"
Caution:  Don't match hyphens in compound words ("well-known") or ranges.
          Require whitespace or start-of-string before the minus.
```

**5.4 — Numeric ranges (digit-dash-digit)**
```
Pattern:  (\d+)\s*-\s*(\d+)(?=\s|$|[,.)!?])
Replace:  "\1 to \2"
Example:  "3-5 items" -> "3 to 5 items"
          "pages 10-20" -> "pages 10 to 20"
Caution:  Must run BEFORE negative number rule. Must not match dates
          (already handled by normalize.h's date patterns). Must not
          match phone numbers (see 5.9).
Priority: Run range detection before negative number detection.
```

**5.5 — Times**
```
Pattern:  (\d{1,2}):(\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.)?
Replace:  Expand to spoken time.
Examples:
  "3:30 PM"  -> "three thirty PM"
  "12:00"    -> "twelve o'clock"
  "9:05 AM"  -> "nine oh five AM"
  "15:30"    -> "fifteen thirty"
  "10:00:45" -> "ten o'clock and forty five seconds" (rare, optional)
Strategy: Hours as number, minutes: if <10 prefix with "oh", "00" -> "o'clock".
          AM/PM kept as literal letters.
```

**5.6 — Fractions**
```
Pattern:  (?<!\d)(\d+)/(\d+)(?!\d)   (not a date like 1/15/2024)
Action:   Convert to spoken fraction:
  1/2 -> "one half"
  1/3 -> "one third"
  1/4 -> "one quarter" (or "one fourth")
  2/3 -> "two thirds"
  3/4 -> "three quarters"
  1/8 -> "one eighth"
  Generic: "\1 over \2" for non-standard fractions
  e.g., 5/7 -> "five over seven" or "five sevenths"
Caution:  Must not match dates. Check that denominator is not a 4-digit year.
          Require that the fraction is not preceded/followed by more digits.
```

**5.7 — Percentages**
```
Already handled in Stage 4.5. Just ensure "percent" is placed after the number
so normalize.h can expand the digit portion.
```

**5.8 — Units and measurements**

Apply when a number is immediately followed by (or separated by a space from)
a known unit abbreviation. Expand the unit to its full spoken form.

```
Singular/Plural rules: use singular when the number is "1", plural otherwise.

Temperature:
  (\d+)\s*°?\s*[Ff]\b  -> "\1 degrees Fahrenheit"
  (\d+)\s*°?\s*[Cc]\b  -> "\1 degrees Celsius"
  (\d+)\s*°\b          -> "\1 degrees"

Length:
  (\d+)\s*mm\b   -> "\1 millimeters"
  (\d+)\s*cm\b   -> "\1 centimeters"
  (\d+)\s*m\b    -> "\1 meters"     (caution: "5m" vs end of sentence "5m.")
  (\d+)\s*km\b   -> "\1 kilometers"
  (\d+)\s*in\b   -> "\1 inches"     (caution: "in" is also a preposition)
  (\d+)\s*ft\b   -> "\1 feet"
  (\d+)\s*yd\b   -> "\1 yards"
  (\d+)\s*mi\b   -> "\1 miles"

Weight:
  (\d+)\s*mg\b   -> "\1 milligrams"
  (\d+)\s*g\b    -> "\1 grams"      (caution: "5g" could be "5G" network)
  (\d+)\s*kg\b   -> "\1 kilograms"
  (\d+)\s*oz\b   -> "\1 ounces"
  (\d+)\s*lb\b   -> "\1 pounds"
  (\d+)\s*lbs\b  -> "\1 pounds"

Volume:
  (\d+)\s*ml\b   -> "\1 milliliters"
  (\d+)\s*mL\b   -> "\1 milliliters"
  (\d+)\s*L\b    -> "\1 liters"

Speed:
  (\d+)\s*mph\b    -> "\1 miles per hour"
  (\d+)\s*km/h\b   -> "\1 kilometers per hour"
  (\d+)\s*kph\b    -> "\1 kilometers per hour"
  (\d+)\s*m/s\b    -> "\1 meters per second"
  (\d+)\s*fps\b    -> "\1 frames per second" or "feet per second" (context)
  (\d+)\s*kbps\b   -> "\1 kilobits per second"
  (\d+)\s*Mbps\b   -> "\1 megabits per second"
  (\d+)\s*Gbps\b   -> "\1 gigabits per second"

Data:
  (\d+)\s*KB\b   -> "\1 kilobytes"
  (\d+)\s*MB\b   -> "\1 megabytes"
  (\d+)\s*GB\b   -> "\1 gigabytes"
  (\d+)\s*TB\b   -> "\1 terabytes"

Electrical:
  (\d+)\s*V\b    -> "\1 volts"
  (\d+)\s*W\b    -> "\1 watts"
  (\d+)\s*kW\b   -> "\1 kilowatts"
  (\d+)\s*A\b    -> "\1 amps"       (caution: "A" is also an article)
  (\d+)\s*mA\b   -> "\1 milliamps"
  (\d+)\s*Hz\b   -> "\1 hertz"
  (\d+)\s*kHz\b  -> "\1 kilohertz"
  (\d+)\s*MHz\b  -> "\1 megahertz"
  (\d+)\s*GHz\b  -> "\1 gigahertz"

Time:
  (\d+)\s*ms\b   -> "\1 milliseconds"
  (\d+)\s*sec\b  -> "\1 seconds"
  (\d+)\s*min\b  -> "\1 minutes"
  (\d+)\s*hr\b   -> "\1 hours"
  (\d+)\s*hrs\b  -> "\1 hours"
```

**Implementation note:** Unit matching must be case-sensitive for many units
(`m` vs `M`, `g` vs `G`, `in` vs `In`). Use a lookup table rather than a
single monster regex. Require a digit before the unit to avoid matching
English words.

**5.9 — Phone numbers**
```
Pattern:  \(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}
Action:   Split into digit groups, spell digit-by-digit.
Example:  "(555) 123-4567" -> "five five five, one two three, four five six seven"
Strategy: Extract digit groups. Speak each group digit-by-digit with commas
          between groups for natural pauses.
Also:     "+1-555-123-4567" -> "plus one, five five five, one two three, four five six seven"
```

**5.10 — Version numbers**
```
Pattern:  v?(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?
Context:  Only when preceded by "version", "v", or at word boundary not in a
          sentence-ending position (not a decimal).
Replace:  "version \1 point \2 point \3 ..."
Example:  "v2.1.3" -> "version 2 point 1 point 3"
          "Python 3.12" -> "Python 3 point 12"
Caution:  Must not match IP addresses or decimal numbers. Require context
          clues (preceding "v", "version", or known software names).
```

**5.11 — Decades**
```
Pattern:  (\d{4})s\b
Replace:  Expand to spoken decade.
Example:  "1990s" -> "nineteen nineties"
          "2010s" -> "twenty tens"
```

**5.12 — Large number suffixes**
```
Pattern:  (\d+(?:\.\d+)?)\s*([KkMmBb])\b
Context:  "K" for thousands, "M" for millions, "B" for billions.
Replace:
  "2.5K"  -> "2.5 thousand"  (normalize.h then expands "2" + "thousand")
  "10M"   -> "10 million"
  "3.2B"  -> "3.2 billion"
Strategy: Replace the suffix with the word, then let decimal/number rules
          handle the numeric part.
Caution:  "K" alone after a non-digit should not match. "MB"/"KB"/"GB" are
          data units handled in 5.8, not this rule.
```

**5.13 — Roman numerals (context-dependent)**
```
Pattern:  \b(I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{0,3}|XII|XIII|XIV|XV)\b
Context:  After names ("Henry VIII"), chapter/section markers ("Chapter IV"),
          or ordinal contexts ("World War II").
Examples:
  "Henry VIII"    -> "Henry the eighth"
  "Chapter IV"    -> "Chapter four"
  "World War II"  -> "World War two"
  "Rocky III"     -> "Rocky three"
Strategy: Maintain a list of contexts (after proper names, after "chapter",
          "part", "section", "act", "episode", "volume", "super bowl").
          Convert Roman numeral to integer, then let normalize.h handle it.
Caution:  "I" as a pronoun must not be matched. Require uppercase and a
          preceding contextual trigger word.
```

---

### Stage 6: Abbreviation Expansion (safety net)

The LLM should already expand abbreviations (Rule 1.8), but these catch any
that leak through. Use a lookup table. Match case-insensitively for some,
case-sensitively for others.

```
Honorifics:
  Dr.     -> "Doctor"
  Mr.     -> "Mister"
  Mrs.    -> "Missus"
  Ms.     -> "Miz"
  Prof.   -> "Professor"
  Jr.     -> "Junior"
  Sr.     -> "Senior"

Location:
  St.     -> "Saint" (default) or "Street" (when preceded by a number)
  Ave.    -> "Avenue"
  Blvd.   -> "Boulevard"
  Rd.     -> "Road"
  Ln.     -> "Lane"
  Ct.     -> "Court"
  Hwy.    -> "Highway"
  Mt.     -> "Mount"

Time:
  Jan. through Dec. -> full month names
  Mon. through Sun. -> full day names
  a.m.    -> "A M"
  p.m.    -> "P M"

Latin:
  e.g.    -> "for example"
  i.e.    -> "that is"
  etc.    -> "et cetera"
  vs.     -> "versus"
  ca.     -> "circa"

Other:
  approx. -> "approximately"
  dept.   -> "department"
  govt.   -> "government"
  Inc.    -> "Incorporated"
  Corp.   -> "Corporation"
  Ltd.    -> "Limited"
  Co.     -> "Company"
  No.     -> "Number" (when before a digit)
  est.    -> "established" (when before a year)
```

---

### Stage 7: Punctuation Normalization

**7.1 — Collapse repeated punctuation**
```
Pattern:  !{2,}     -> "!"
Pattern:  \?{2,}    -> "?"
Pattern:  \.{4,}    -> "..."
Pattern:  ,{2,}     -> ","
Pattern:  !+\?+|\?+!+  -> "?"   (mixed excl/question -> question)
```

**7.2 — Semicolons to commas**
```
Pattern:  ;
Replace:  ,
Reason:   TTS engines handle comma pauses better than semicolons. Kokoro
          does have a semicolon in its vocabulary (token 1) but commas
          produce more natural prosody for conversational speech.
```

**7.3 — Mid-sentence colons to commas**
```
Pattern:  (?<=\w)\s*:\s*(?=\w)     (colon between words, not in times)
Replace:  ", "
Example:  "The answer: forty two" -> "The answer, forty two"
Caution:  Must not match times like "3:30" (already handled in Stage 5.5
          which runs before this). Use negative lookbehind for digits:
          (?<!\d):(?!\d)
```

**7.4 — Double/triple dashes**
```
Pattern:  -{2,}
Replace:  ", "
Example:  "wait -- what?" -> "wait, what?"
Note:     normalize.h converts em-dash to "--". This catches those and any
          raw dashes in the input.
```

**7.5 — Slash as "or" in word contexts**
```
Already handled in Stage 4.13. Verify no remaining slashes except in
deliberately kept contexts.
```

**7.6 — Stacked/misplaced punctuation cleanup**
```
Pattern:  ,\s*\.   -> "."
Pattern:  \.\s*,   -> "."
Pattern:  ,\s*,    -> ","
Pattern:  \s+([.,!?])  -> "\1"   (remove space before punctuation)
```

---

### Stage 8: Whitespace and Final Cleanup

**8.1 — Normalize line breaks**
```
Replace \r\n and \r with \n.
Replace multiple consecutive \n with a single space.
(Spoken text has no "paragraphs" — it's all one stream.)
```

**8.2 — Collapse multiple spaces**
```
Pattern:  \s{2,}
Replace:  " " (single space)
```

**8.3 — Trim**
```
Strip leading and trailing whitespace.
```

**8.4 — Ensure terminal punctuation**
```
If the text does not end with . ! or ?, append a period.
Reason:   Terminal punctuation signals to the TTS model that the utterance
          is complete, affecting the final intonation contour. Without it,
          the last word may sound clipped or have rising intonation.
```

**8.5 — Strip leading/trailing non-alphanumeric junk**
```
After all processing, there may be stray leading commas, spaces, or
punctuation. Trim any leading punctuation that isn't a quote or parenthesis.
Pattern:  ^[,;:\s]+
Replace:  (empty string)
```

---

### Stage 9: Sentence Splitting (for streaming / latency optimization)

Rokoko supports streaming synthesis via `/synthesize/stream`. Splitting text
into sentences and synthesizing them individually reduces time-to-first-audio.
The current chunking logic in `main.cu` (`chunk_ipa()`) splits at the IPA
level after G2P. Sentence-level splitting should happen BEFORE G2P for optimal
streaming.

**9.1 — Split points**
```
Split after: . ! ? followed by whitespace and an uppercase letter (or end of string).
Regex:       (?<=[.!?])\s+(?=[A-Z])
```

**9.2 — Do NOT split on these false positives**

Abbreviation periods (maintain a no-split set):
```
Mr. Mrs. Ms. Dr. Prof. Jr. Sr. St. Ave. Blvd.
Jan. Feb. Mar. Apr. Jun. Jul. Aug. Sep. Oct. Nov. Dec.
Mon. Tue. Wed. Thu. Fri. Sat. Sun.
a.m. p.m. e.g. i.e. vs. etc. approx. dept. govt.
Inc. Corp. Ltd. Co. No. Mt.
U.S. U.K. E.U. U.N.
```

Decimal numbers:
```
"3.14" — the period is a decimal point, not a sentence end.
Regex:  \d\.\d should not trigger a split.
```

Ellipsis:
```
"..." — three dots indicate a pause, not three sentence ends.
Collapse to a single ellipsis token before splitting.
```

Initials:
```
"J. K. Rowling" — periods after single capital letters are initials.
Pattern:  [A-Z]\. [A-Z]\.  should not split.
```

**9.3 — Streaming strategy**

For each sentence:
1. Run through deterministic rules (Stages 1-8)
2. Run through `text_norm::preprocess_text()` (money, dates, numbers, unicode)
3. Run G2P inference
4. Run TTS inference
5. Stream audio chunk to client

This allows audio for sentence 1 to start playing while sentence 2 is still
being processed by G2P/TTS.

---

## Processing Order Summary

```
Input text (from LLM)
  |
  +--[Stage 1]  Strip surviving markup (markdown, HTML, lists)
  +--[Stage 2]  Unicode normalization (smart quotes, invisibles, symbols)
  +--[Stage 3]  Arrow/sequence symbols -> words
  +--[Stage 4]  Symbol verbalization (& @ # % ^ | \ _ () {} [] ~ *)
  +--[Stage 5]  Quantities (ordinals, decimals, negatives, ranges, times,
  |              fractions, units, phones, versions, decades, roman numerals)
  +--[Stage 6]  Abbreviation expansion (safety net)
  +--[Stage 7]  Punctuation normalization (collapse, semicolons, colons, dashes)
  +--[Stage 8]  Whitespace cleanup and terminal punctuation
  |
  v
[normalize.h]   Money -> Dates -> Standalone numbers -> Unicode->ASCII
  |
  v
[Sentence split] -> per-sentence G2P -> TTS -> stream audio
```

---

## Integration Points

### Where to add the new code

Create a new header: `src/speech_prep.h`

```cpp
#pragma once
#include <string>

namespace speech_prep {

// Run all deterministic preprocessing stages (1-8).
// Call this BEFORE text_norm::preprocess_text().
std::string prepare_for_tts(const std::string& text);

// Split text into sentences for streaming synthesis.
// Call AFTER prepare_for_tts() and text_norm::preprocess_text().
std::vector<std::string> split_sentences(const std::string& text);

} // namespace speech_prep
```

### Where to call it

In `main.cu`, the `TtsPipeline::synthesize_streaming()` method currently does:

```cpp
std::string preprocessed = text_norm::preprocess_text(text);
```

Change to:

```cpp
std::string prepared = speech_prep::prepare_for_tts(text);
std::string preprocessed = text_norm::preprocess_text(prepared);
```

For sentence-level streaming, the flow becomes:

```cpp
std::string prepared = speech_prep::prepare_for_tts(text);
std::string preprocessed = text_norm::preprocess_text(prepared);
auto sentences = speech_prep::split_sentences(preprocessed);
for (auto& sentence : sentences) {
    std::string ipa = g2p.infer(sentence, ltHandle, stream);
    // ... chunk and synthesize ...
}
```

The same change applies to the CLI mode code path.

### Implementation constraints

- **Single-header C++17**, matching the style of `normalize.h`.
- **No external dependencies**. Use `<regex>`, `<string>`, `<vector>`,
  `<unordered_map>` from the standard library.
- **Pure functions**. No global state, no allocations beyond strings.
- **UTF-8 aware**. Input may contain multi-byte characters. Use the UTF-8
  helpers already in `normalize.h` (or copy them).

---

## Test Cases

Every rule should be tested. Here are key test vectors:

### Markup stripping
```
Input:    "This is **bold** and *italic* and `code`"
Expected: "This is bold and italic and code"

Input:    "See [the docs](https://example.com) for details."
Expected: "See the docs for details."

Input:    "## Heading\n- item 1\n- item 2"
Expected: "Heading\nitem 1\nitem 2"
          (whitespace cleanup then produces: "Heading item 1 item 2")
```

### Arrows
```
Input:    "Move -> Jump -> Attack"
Expected: "Move then Jump then Attack"

Input:    "A => B => C"
Expected: "A then B then C"
```

### Symbols
```
Input:    "Try queue_entry() and see if that works"
Expected: "Try queue entry and see if that works"

Input:    "R&D is important"
Expected: "R and D is important"

Input:    "2 + 2 = 4"
Expected: "2 plus 2 equals 4"

Input:    "x^2 + y^2"
Expected: "x squared plus y squared"

Input:    "~100 users"
Expected: "approximately 100 users"
```

### Numbers and quantities
```
Input:    "The 3rd floor"
Expected: "The third floor"

Input:    "3.14 is pi"
Expected: "3 point 1 4 is pi"
          (after normalize.h: "three point one four is pi")

Input:    "-5 degrees"
Expected: "negative 5 degrees"
          (after normalize.h: "negative five degrees")

Input:    "3-5 items"
Expected: "3 to 5 items"

Input:    "3:30 PM"
Expected: "three thirty PM"

Input:    "3/4 of the way"
Expected: "three quarters of the way"

Input:    "50%"
Expected: "50 percent"
          (after normalize.h: "fifty percent")

Input:    "5kg"
Expected: "5 kilograms"
          (after normalize.h: "five kilograms")

Input:    "(555) 123-4567"
Expected: "five five five, one two three, four five six seven"

Input:    "v2.1.3"
Expected: "version 2 point 1 point 3"
```

### Punctuation
```
Input:    "Wait -- what?!"
Expected: "Wait, what?"

Input:    "The answer: yes."
Expected: "The answer, yes."

Input:    "Really?!?!"
Expected: "Really?"
```

### Abbreviations
```
Input:    "Dr. Smith on 5th Ave."
Expected: "Doctor Smith on 5th Avenue."

Input:    "e.g. this works"
Expected: "for example this works"
```

### End-to-end
```
Input:    "**Note:** The temperature is ~72°F (±2°F) -- see https://weather.com for details!"
Expected: "Note, The temperature is approximately 72 degrees Fahrenheit, plus or minus 2 degrees Fahrenheit, see the weather website for details!"
          (after normalize.h expands numbers: "Note, The temperature is approximately seventy two degrees Fahrenheit, plus or minus two degrees Fahrenheit, see the weather website for details!")
```

### Sentence splitting
```
Input:    "Hello world. How are you? I'm fine."
Split:    ["Hello world.", "How are you?", "I'm fine."]

Input:    "Dr. Smith arrived at 3 p.m. He was late."
Split:    ["Dr. Smith arrived at 3 p.m.", "He was late."]
          (NOT split after "Dr." or "p.m.")

Input:    "The value is 3.14. That's pi."
Split:    ["The value is 3.14.", "That's pi."]
          (NOT split after "3.")
```

---

## Edge Cases and Cautions

1. **Rule ordering conflicts**: Ranges (`3-5`) vs. negative numbers (`-5`) vs.
   hyphens (`well-known`). Ranges must be detected first, then negatives, and
   hyphens should not be touched.

2. **Date vs. fraction**: `3/4` could be March 4th or three-quarters.
   `normalize.h` requires `M/D/YYYY` (4-digit year) for dates, so bare `3/4`
   will not be caught by it. The fraction rule can safely claim `digit/digit`
   patterns where the denominator is 1-2 digits.

3. **Units vs. words**: `"5 in the morning"` — `in` is a preposition, not
   inches. Require NO space between digit and unit for ambiguous short units
   (`in`, `m`, `g`, `A`), or only match when unit is immediately adjacent to
   the digit (`5in` not `5 in`).

4. **"$" in code/shell contexts**: The LLM should not include code, but if it
   leaks through, `$VAR` should not be treated as currency. normalize.h
   requires `$` + digit, so non-digit `$` patterns are safe.

5. **Performance**: `std::regex` in C++ can be slow for complex patterns. For
   hot paths, prefer manual string scanning (like normalize.h does for money
   and numbers) over regex. Regex is fine for one-shot patterns that match
   rarely.

6. **Idempotency**: Running the pipeline twice on the same text should produce
   the same result. Ensure rules don't create patterns that trigger other rules
   incorrectly (e.g., "three quarters" should not be re-processed).
