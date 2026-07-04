(define (problem reach-goal-problem)
  (:domain reach-goal-domain)

  (:objects
    avatar goal - object
  )

  (:init
    ;; Initially, the avatar has not reached the goal
    (not (reached avatar goal))
  )

  (:goal
    ;; The avatar needs to reach the goal
    (reached avatar goal)
  )
)