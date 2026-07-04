(define (problem minotaur-problem)
  (:domain minotaur-domain)

  (:objects
    wall minotaur staircase_down agent wand inventory - object
  )

  (:init
    (not (killed agent minotaur))
  )

  (:goal
    (killed agent minotaur)
  )
)