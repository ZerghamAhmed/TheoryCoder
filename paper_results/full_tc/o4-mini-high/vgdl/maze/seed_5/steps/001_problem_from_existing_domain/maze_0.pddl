(define (problem gridgame-problem)
  (:domain gridgame-domain)

  (:objects
    avatar cheese - object
  )

  (:init
    ;;; Initially the avatar is not overlapping the cheese
    (not (overlaps avatar cheese))
  )

  (:goal
    (overlaps avatar cheese)
  )
)