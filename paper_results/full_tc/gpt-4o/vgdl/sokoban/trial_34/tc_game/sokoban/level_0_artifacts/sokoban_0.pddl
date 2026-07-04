(define (problem box-to-hole-problem)
  (:domain reach-goal-domain)

  (:objects
    avatar box hole - object
  )

  (:init
    ;; Initially, the box has not reached the hole
    (not (reached box hole))
  )

  (:goal
    ;; Goal is for the box to reach the hole
    (reached box hole)
  )
)