(define (problem grid-problem)
  (:domain grid-domain)

  (:objects
    avatar box hole - object
  )

  (:init
    ;; Initially, nothing has been "reached."
    (not (reached box hole))
    (not (reached avatar box))
  )

  (:goal
    ;; Goal: ensure the box is aligned with the hole
    (reached box hole)
  )
)