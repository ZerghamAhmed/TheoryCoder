(define (problem gridgame-problem)
  (:domain gridgame-domain)

  (:objects
    red_agent blue_ball grey_wall agent_direction agent_carrying - object
  )

  (:init
    ;; agent is not holding the blue_ball initially
    (not (holding red_agent blue_ball))
  )

  (:goal
    (holding red_agent blue_ball)
  )
)
