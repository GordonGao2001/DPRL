from random import randrange
import matplotlib.pyplot as plt
inventory = [20, 20]
def policy (inv): #return inventory, hold_cost, order_cost

    if inv[0] > 1 and inv[1] > 1:
        match int(randrange(3)):
            case 0:
                i1 = inv[0] - 1
                i2 = inv[1]
                return [i1, i2], i1+i2*2 , 0
            case 1:
                i1 = inv[0]
                i2 = inv[1] - 1
                return [i1,i2], i1+i2*2 , 0
            case 2:
                i1 = inv[0] - 1
                i2 = inv[1] - 1
                return [i1, i2], i1+i2*2 , 0
            case _:
                return inv, inv[0]+inv[1]*2, 0
    else:
        return [5, 5], 15 ,5

t = 100
t2 = 10000000
hd_cost=0
ord_cost=0
ttl_cost = 0
hd_cost_l = []
ord_cost_l = []
inv1 = []
inv2 = []
ttl_cost_l =[]
avg_cost_l=[]
for i in range(t):
    tmp_inventory, tmp_hd_cost , tmp_ord_cost = policy (inventory)
    hd_cost_l.append(tmp_hd_cost)
    ord_cost_l.append(tmp_ord_cost)
    inv1.append(tmp_inventory[0])
    inv2.append(tmp_inventory[1])
    hd_cost += tmp_hd_cost
    ord_cost += tmp_ord_cost
    ttl_cost += tmp_hd_cost + tmp_ord_cost
    ttl_cost_l.append(ttl_cost)
    if i == 0:
        avg_cost = ttl_cost / 1
    else:
        avg_cost = ttl_cost / i
    ttl_cost_l.append(ttl_cost)
    avg_cost_l.append(avg_cost)
    inventory = tmp_inventory
    # print(inventory)
    # print(hd_cost)
    # print(ord_cost)

plt.figure(figsize=(15, 10))

# Holding and Ordering Costs over Time
fig1=plt.subplot(4, 1, 1)
fig1.set_xticks(range(0, t, 3))
plt.plot(range(t), hd_cost_l, label="Holding Cost", alpha=0.8)
plt.plot(range(t), ord_cost_l, label="Ordering Cost", alpha=0.8)
plt.plot(range(t), avg_cost_l, label="Avg. Cost", alpha=0.8)
plt.title("Costs Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Cost")
fig1.text(t - 1, avg_cost_l[-1], f"{avg_cost_l[-1]:.2f}", color="blue", fontsize=10, va="center")
plt.legend()
plt.subplot(4, 1, 2).set_xticks(range(0, t, 3))
plt.plot(range(t), inv1, label="Inventory Product 1", alpha=0.8)

plt.plot(range(t), inv2, label="Inventory Product 2", alpha=0.8)

plt.title("Inventory Levels Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Inventory Level")
plt.legend()

inventory = [5,5]

hd_cost=0
ord_cost=0
ttl_cost = 0
hd_cost_l = []
ord_cost_l = []
inv1 = []
inv2 = []
ttl_cost_l =[]
avg_cost_l=[]
for i in range(t2):
    tmp_inventory, tmp_hd_cost , tmp_ord_cost = policy (inventory)
    hd_cost_l.append(tmp_hd_cost)
    ord_cost_l.append(tmp_ord_cost)
    inv1.append(tmp_inventory[0])
    inv2.append(tmp_inventory[1])
    hd_cost += tmp_hd_cost
    ord_cost += tmp_ord_cost
    ttl_cost += tmp_hd_cost+tmp_ord_cost
    if i == 0:
        avg_cost = ttl_cost/1
    else:
        avg_cost = ttl_cost / i
    ttl_cost_l.append(ttl_cost)
    avg_cost_l.append(avg_cost)
    inventory = tmp_inventory

# Holding and Ordering Costs over Time
fig2=plt.subplot(4, 1, 3)
plt.plot(range(0,200), hd_cost_l[:200], label="Holding Cost", alpha=0.8)
plt.plot(range(0,200), ord_cost_l[:200], label="Ordering Cost", alpha=0.8)
plt.plot(range(0,200), avg_cost_l[:200], label="Avg. Cost", alpha=0.8)
plt.title("Costs Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Cost")

fig2.text(t - 1, avg_cost_l[-1], f"{avg_cost_l[-1]:.2f}", color="blue", fontsize=10, va="center")
plt.legend()
plt.subplot(4, 1,4)
plt.plot(range(0,200), inv1[:200], label="Inventory Product 1", alpha=0.8)
plt.plot(range(0,200), inv2[:200], label="Inventory Product 2", alpha=0.8)
print("***")
print(sum(inv1)/t)
print(sum(inv2)/t)
plt.title("Inventory Levels Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Inventory Level")
plt.legend()

plt.tight_layout()

plt.savefig("t_100_and_inv_20_20_t_1000_inv_5_5.png", dpi=300, bbox_inches='tight')
plt.show()


pows = input("input the 0-3 to simulate the real avg. , 1 is fine, 3 is really slow ")
pows = int(pows)
if pows not in range(4):
    print("invalid!!")
itr = 100*pow(10, pows)

for j in range(itr):
    t = 100
    t2 = 200
    hd_cost = 0
    ord_cost = 0
    ttl_cost = 0
    hd_cost_l = []
    ord_cost_l = []
    inv1 = []
    inv2 = []
    ttl_cost_l = []
    avg_cost_l = []
    avg_avg = []
    inventory = [5, 5]
    for i in range(t*10):
        tmp_inventory, tmp_hd_cost , tmp_ord_cost = policy (inventory)
        hd_cost_l.append(tmp_hd_cost)
        ord_cost_l.append(tmp_ord_cost)
        inv1.append(tmp_inventory[0])
        inv2.append(tmp_inventory[1])
        hd_cost += tmp_hd_cost
        ord_cost += tmp_ord_cost
        ttl_cost += tmp_hd_cost + tmp_ord_cost
        ttl_cost_l.append(ttl_cost)
        if i == 0:
            avg_cost = ttl_cost / 1
        else:
            avg_cost = ttl_cost / i
        ttl_cost_l.append(ttl_cost)
        avg_cost_l.append(avg_cost)
        inventory = tmp_inventory
    avg_avg.append(avg_cost_l.pop(len(avg_cost_l)-1))
print("the avg. cost is "+str(sum(avg_avg)/len(avg_avg))+f" with inventory starting from ({inventory[0]},{inventory[1]}) with {itr} steps")