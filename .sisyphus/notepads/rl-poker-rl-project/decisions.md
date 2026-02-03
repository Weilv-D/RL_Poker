# Decisions

## Architectural Choices

(Initial notepad created - no decisions yet)

## Task 2: Rules Engine Decisions

### Rank Representation
- **Decision**: Use `IntEnum` for Rank with explicit integer values
- **Rationale**: Allows natural comparison with Python operators while maintaining type safety

### Hand as Immutable Dataclass
- **Decision**: `Hand` is a frozen dataclass with `FrozenSet[Card]` for cards
- **Rationale**: Hands are conceptually immutable; frozen allows hashing for potential caching

### Separate main_rank vs secondary_rank
- **Decision**: Hand stores `main_rank` (for comparison) and optional `secondary_rank` (kickers)
- **Rationale**: Comparison logic only uses main_rank; secondary_rank preserved for potential display/debug

### parse_hand Returns Optional Instead of Raising
- **Decision**: `parse_hand()` returns `None` for invalid combinations rather than raising exceptions
- **Rationale**: Invalid combinations are expected in move enumeration; exceptions would be expensive

### No Tail-Hand Exemptions in Rules Module
- **Decision**: Rules module handles standard hand parsing only; exemptions belong in moves module
- **Rationale**: Separation of concerns - rules define what hands ARE, moves define what hands can be PLAYED

## Task 3: Legal Moves + Action Encoding Decisions

### MAX_ACTIONS = 1000
- **Decision**: Set MAX_ACTIONS to 1000 instead of initial 500
- **Rationale**: Testing revealed 13-card hands with quads generate 715+ legal moves; 1000 provides margin

### Move as Frozen Dataclass
- **Decision**: `Move` is frozen dataclass with FrozenSet[Card] for hashability
- **Rationale**: Moves need to be hashable for dict mappings in ActionSpace

### Exemption Tracking via Flags
- **Decision**: Use `is_exemption` flag and `standard_type` field rather than special HandType
- **Rationale**: Exemptions represent standard hand types played with fewer cards; separating flags keeps Hand logic clean

### Overflow Guard
- **Decision**: Raise explicit error rather than silently truncating legal moves
- **Rationale**: Truncation would silently remove valid actions, violating RL correctness guarantees

### Context Pattern for Legal Moves
- **Decision**: Use `MoveContext` dataclass to pass game state (previous_move, is_tail_hand, etc.)
- **Rationale**: Clean interface, easy to extend with additional context fields
