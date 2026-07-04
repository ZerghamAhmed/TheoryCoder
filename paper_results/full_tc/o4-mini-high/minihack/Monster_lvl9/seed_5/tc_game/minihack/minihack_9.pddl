(define (problem rogue-problem)
  (:domain rogue-domain)

  (:objects
    human_rogue_called_agent newt fox lichen staircase_down - object
  )

  (:init
    ;; initially no overlapping facts are true
  )

  (:goal
    (and
      ;; primary mission: descend the stairs
      (overlap human_rogue_called_agent staircase_down)
      ;; subgoals: avoid bumping into creatures
      (not (overlap human_rogue_called_agent newt))
      (not (overlap human_rogue_called_agent fox))
      (not (overlap human_rogue_called_agent lichen))
    )
  )
)