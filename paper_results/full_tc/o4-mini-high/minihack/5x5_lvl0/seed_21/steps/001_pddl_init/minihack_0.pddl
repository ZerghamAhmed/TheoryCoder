(define (problem rogue-problem)
  (:domain rogue-domain)

  (:objects
    human_rogue_called_agent staircase_down staircase_up inventory won lost - object
  )

  (:init
    (not (descended human_rogue_called_agent staircase_down))
  )

  (:goal
    (descended human_rogue_called_agent staircase_down)
  )
)