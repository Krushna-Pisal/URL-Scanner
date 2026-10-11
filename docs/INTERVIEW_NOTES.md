# Interview Notes

## Why group the train/test split by domain?

URLs from the same site often share hostname spelling, subdomains, paths, and query patterns. A random row split can put one URL from a domain in training and another URL from the same domain in testing. The model then gets a familiar domain pattern in the test set, which can make it look like it handles new sites better than it really does. `GroupShuffleSplit` keeps each approximate registered-domain group on one side of the split. This project uses a simple last-two-label approximation, so suffixes such as `co.in` are a known limitation.

## What does stacking do?

Stacking trains multiple base models, here logistic regression and random forest. Their predictions become inputs to a final logistic regression model. The final model learns how to combine those predictions. It can benefit when the base models make different kinds of mistakes, but it is more complex and is not guaranteed to perform better. The held-out domain test set is used to compare it with each simpler model.

## Precision, recall, and false-positive rate

- **Precision** asks: among URLs flagged as phishing, what share really has label 1? Low precision means more legitimate URLs are blocked or warned about.
- **Recall** asks: among phishing URLs, what share did the model find? Low recall means more phishing URLs pass through.
- **False-positive rate (FPR)** asks: among legitimate URLs, what share was incorrectly flagged? For a URL filter, FPR matters because a legitimate URL warning or block interrupts normal work and can reduce trust in the filter. Precision and FPR answer related but different questions because precision also depends on how common phishing is in the evaluated sample.

The project reports all three because a filter needs to balance missed attacks with the cost of blocking safe traffic.

## Why do not fetch the URL?

The scanner is deliberately string-only. Fetching an attacker-controlled URL from the scanner could let it reach internal services or metadata endpoints, creating an SSRF risk. Fetching also adds latency, makes outcomes depend on network state and page changes, and can expose the scanner to hostile content. Character-level analysis is repeatable for the same input and model and does not need outbound requests.

## What is ablation?

The training script finds the most important feature for the selected model, removes that one feature, and trains the selected model again on the same grouped split. It reports the new metrics and the difference from the full-feature run. This checks whether performance depends heavily on one signal. A large drop suggests that signal contributes substantially; a small change suggests the remaining features can provide similar information. Ablation is diagnostic, not proof that a feature causes the outcome.

## How does Levenshtein distance work?

Levenshtein distance is the minimum number of single-character insertions, deletions, or substitutions needed to change one string into another. The implementation uses dynamic programming: each table cell stores the cheapest edit cost for two prefixes. A cell compares the cost of inserting, deleting, or matching/substituting the next character. This makes lookalike spellings measurable without a third-party distance package.

## Dataset-bias risk

The model learns from the examples and labels it receives. If phishing and legitimate examples come from different sources, time periods, or collection methods, it can learn those collection patterns instead of useful threat signals. Duplicate domains across a random split can also inflate results, which is why the project groups domains. Grouping does not fix bad labels, a narrow sample, source-specific patterns, or the approximate domain parser. High scores should be checked against the data and split before making security claims.
