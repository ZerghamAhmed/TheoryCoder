(define (domain babyai)
    (:requirements :strips :negative-preconditions :typing)

    (:types 
        object door
    )

    (:predicates
        (controllable ?obj - object)
        (overlapping ?obj - object ?to - object)
        (carrying ?obj - object)
        (open_door ?door - door)
        (locked ?door - door)
        (next_to ?obj1 - object ?obj2 - object)
        (unlocks ?key - object ?door - door)
        (blocking ?obj - object ?door - door)
        (clear ?door - door)
        (inventory_full)
        (agent_moved_away ?door - door)
        (not_near_door ?obj)
        (found ?obj - object)
        (unfound ?obj - object)
        (went_through ?agent ?door)

    )

    (:action move_to
        :parameters (?obj - object ?to - object)
        :precondition (and (controllable ?obj) (not (next_to ?obj ?to)))
        :effect (next_to ?obj ?to)
    )

    (:action unblock
        :parameters (?door - door ?obj - object)
        :precondition (and (blocking ?obj ?door) (not (clear ?door)) (not (carrying ?obj)) (not (inventory_full)))
        :effect (and (agent_moved_away ?door) (not (blocking ?obj ?door)) (clear ?door) (carrying ?obj) (inventory_full))
    )
    
    (:action drop
        :parameters (?obj - object)
        :precondition (and (carrying ?obj) (inventory_full))
        :effect (and (not (carrying ?obj)) (not (inventory_full)) (not_near_door ?obj))
    )

    (:action pickup
        :parameters (?item - object)
        :precondition (not (carrying ?item))
        :effect (carrying ?item)
    )

    (:action put_next_to
        :parameters (?item - object ?adjacent - object)
        :precondition (not (next_to ?item ?adjacent))
        :effect (next_to ?item ?adjacent)
    )

    (:action open
        :parameters (?door - door)
        :precondition (and (not (open_door ?door)) (not (locked ?door)))
        :effect (open_door ?door)
    )

    (:action open_box_for_key
        :parameters (?box - object ?key - object)
        :precondition (and (not (found ?key)))
        :effect (found ?key)
    )

    (:action unlock
        :parameters (?door - door ?key - object)
        :precondition (and (locked ?door) (unlocks ?key ?door) (clear ?door))
        :effect (open_door ?door)
    )
)