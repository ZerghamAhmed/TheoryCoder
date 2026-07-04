(define (problem reachgame-problem)
  (:domain reachgame)

  (:objects
    avatar box hole wall floor - object
  )

  (:init
    ;; initially, no two objects are colocated
  )

  (:goal
    (colocated box hole)
  )
)