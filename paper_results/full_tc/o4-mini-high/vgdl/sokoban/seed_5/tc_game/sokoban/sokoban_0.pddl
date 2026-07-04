(define (problem gridgame-problem)
  (:domain gridgame-domain)

  (:objects
    avatar box hole wall floor - object
  )

  (:init
    ;; nothing overlaps initially
  )

  (:goal
    ;; win when the box overlaps the hole
    (overlaps box hole)
  )
)