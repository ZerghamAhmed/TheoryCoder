(define (problem pickup-problem)
  (:domain pickup-domain)

  (:objects
    red_agent   - agent
    yellow_key  - key
  )

  (:init
    (not (has red_agent yellow_key))
  )

  (:goal
    (has red_agent yellow_key)
  )
)