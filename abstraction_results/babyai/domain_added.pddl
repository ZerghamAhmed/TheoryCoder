(define (domain minimal-game)
(:requirements :strips :typing)

(:types
object
)

(:predicates
has ?a - object ?b - object
unlocks ?a - object ?b - object
islocked ?x - object
)

(:action pickup
:parameters (?a - object ?b - object)
:precondition (not (has ?a ?b))
:effect (has ?a ?b)
)

(:action unlock
:parameters (?a - object ?b - object)
:precondition (and (has ?a ?b) (islocked ?b))
:effect (not (islocked ?b))
)
)
