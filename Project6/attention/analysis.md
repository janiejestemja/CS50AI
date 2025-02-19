# Analysis

## Layer 4, Head 1

This attention head appears to relate the same word with itself, and otherweise to reappearences of the same word in the sentence (if any). This results in a symmetrical diagram.

Example Sentences:
- If penguins are birds and birds can fly then penguins can [MASK].
- If penguins are birds and birds can fly then penguins [MASK] fly.
- If penguins are birds and birds can fly then [MASK] can fly.

## Layer 5, Head 4

For the most part this attention head appears to act very similar to Layer 4, Head 1 but except for relations to identical words this attention head appears to also have learned *conditionals* additionaly by establishing a connection between `then` and `if`.

Example Sentences:
- If penguins are birds and birds can fly then penguins can [MASK].
- If penguins are birds and birds can fly then penguins [MASK] fly.
- If penguins are birds and birds can fly then [MASK] can fly.
