(define (problem reach-goal-problem)
  (:domain reach-goal-domain)

  (:objects
    avatar - object
    cheese - object
  )

  (:init
    ;; Initially the goal (cheese) has not been reached by the avatar
    (not (reached avatar cheese))
  )

  (:goal
    (reached avatar cheese)
  )
)