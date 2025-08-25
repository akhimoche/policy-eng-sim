from utils.norms.growth_reserve_local import GrowthReserveLocal

norm = GrowthReserveLocal(radius=2.0, K=3, penalty=5.0, seed=0)
norm.set_epsilon("agent_0", 0.2)
norm.update_apples({(5,5), (5,6), (6,5), (7,8)})

print("Penalty for (5,4)->(5,5):", norm.step_penalty("agent_0", (5,4), (5,5)))
print("Penalty for (0,0)->(0,1):", norm.step_penalty("agent_0", (0,0), (0,1)))
