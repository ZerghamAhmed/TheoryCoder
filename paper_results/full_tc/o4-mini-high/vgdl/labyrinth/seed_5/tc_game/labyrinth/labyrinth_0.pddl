(define (problem gridgame-problem)
  (:domain gridgame-domain)
  (:objects
    avatar goal - object
  )
  (:init
    ;; initially the avatar does not overlap the goal
    (not (overlaps avatar goal))
  )
  (:goal
    ;; the avatar must overlap the goal
    (overlaps avatar goal)
  )
)