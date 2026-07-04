(define (problem dungeon-problem)
  (:domain dungeon)

  (:objects
    staircase_down jackal newt human_rogue_called_agent - object
  )

  (:init
    (overlap human_rogue_called_agent staircase_down)
  )

  (:goal
    (won)
  )
)