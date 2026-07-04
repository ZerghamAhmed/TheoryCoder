(define (problem sokoban-problem)
  (:domain sokoban-domain)
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