(define (problem sokoban-problem)
  (:domain sokoban)
  (:objects
    box hole - object
  )
  (:init
    (not (overlaps box hole))
  )
  (:goal
    (overlaps box hole)
  )
)