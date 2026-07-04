(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human agent staircase - object
  )

  (:init
    ; initially, the human has not descended the staircase
    (not (descended human))
  )

  (:goal
    (descended human)
  )
)