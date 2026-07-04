(define (problem game-problem)
  (:domain game-domain)
  (:objects
    avatar cheese - object
  )
  (:init
    (not (reaches avatar cheese))
  )
  (:goal
    (reaches avatar cheese)
  )
)