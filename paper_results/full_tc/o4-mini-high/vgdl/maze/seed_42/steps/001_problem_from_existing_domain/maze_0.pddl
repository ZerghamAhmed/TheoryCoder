(define (problem grid-problem)
  (:domain grid-domain)

  (:objects
    avatar cheese floor - object
  )

  (:init
    ;; no initial reaches relations; everything is implicitly false
  )

  (:goal
    (and
      (reaches avatar floor)
      (reaches avatar cheese)
    )
  )
)