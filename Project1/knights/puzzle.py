from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # rules
    Or(AKnave, AKnight),
    Not(And(AKnave, AKnight)),

    # what a said
    Implication(AKnight, And(AKnave, AKnight))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # rules
    Or(AKnave, AKnight),
    Not(And(AKnave, AKnight)),

    Or(BKnave, BKnight),
    Not(And(BKnave, BKnight)),

    # what a said
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2
# A says "We are the same kind.")
# B says "We are of different kinds."
knowledge2 = And(
    # rules
    Or(AKnave, AKnight),
    Not(And(AKnave, AKnight)),

    Or(BKnave, BKnight),
    Not(And(BKnave, BKnight)),

    # what a said
    Implication(AKnight, Or(And(AKnave, BKnave), And(AKnight, BKnight))),
    Implication(AKnave, Not(Or(And(AKnave, BKnave), And(AKnight, BKnight)))),

    # what b said
    Implication(BKnight, Or(And(AKnave, BKnight), And(AKnight, BKnave))),
    Implication(BKnave, Not(Or(And(AKnave, BKnight), And(AKnight, BKnave))))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # rules
    Or(AKnave, AKnight),
    Not(And(AKnave, AKnight)),

    Or(BKnave, BKnight),
    Not(And(BKnave, BKnight)),

    Or(CKnave, CKnight),
    Not(And(CKnave, CKnight)),

    # what a said
    Implication(
        AKnight, 
        Or(
            AKnave, 
            AKnight
        )
    ),
    Implication(
        AKnave, 
        Or(
            AKnave, 
            AKnight
        )
    ),

    # what b said
    Implication(
        BKnight,
            And(
                Implication(
                AKnight,
                BKnave,
                ),
                CKnave,
            )
    ),
    Implication(
        BKnave,
        Not(And(
            Implication(
                AKnight,
                BKnave,
            ),
            CKnave,
        ))
    ),


    # what c said
    Implication(
        CKnight,
        AKnight
    ),
    Implication(
        CKnave,
        AKnight
    ),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
