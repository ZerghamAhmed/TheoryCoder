(define (problem pickup-problem)
  (:domain pickup-domain)
  (:objects
    red_agent   - agent
    yellow_key  - key
  )
  (:init
    (not (carrying red_agent yellow_key))
  )
  (:goal
    (carrying red_agent yellow_key)
  )
)