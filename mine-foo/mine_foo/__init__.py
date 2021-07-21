from gym.envs.registration import register

register(
    id='three-cars-v0',
    entry_point='mine_foo.envs:ThreeCars',
)

register(
    id='three-cars-hard-v0',
    entry_point='mine_foo.envs:ThreeCarsHard',
)

register(
    id='three-cars-pcgrl-v0',
    entry_point='mine_foo.envs:ThreeCarsPCGRL',
)

register(
    id='three-cars-pcgrl-RLlib-v0',
    entry_point='mine_foo.envs:ThreeCarsPCGRL-RLlib',
)