(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human_rogue_called_agent jackal newt staircase_down - object
  )

  (:init
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)