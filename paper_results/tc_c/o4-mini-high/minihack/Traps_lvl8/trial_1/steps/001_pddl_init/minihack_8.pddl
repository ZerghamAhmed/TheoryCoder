(define (problem dungeon-problem)
  (:domain dungeon-domain)

  (:objects
    human_rogue_called_agent staircase_down - object
  )

  (:init
    (not (won human_rogue_called_agent))
  )

  (:goal
    (won human_rogue_called_agent)
  )
)