(define (problem pickup-problem)
  (:domain pickup-domain)

  (:objects
    red_agent    - agent
    yellow_key   - key
  )

  (:init
    (not (pickedup red_agent yellow_key))
  )

  (:goal
    (pickedup red_agent yellow_key)
  )
)