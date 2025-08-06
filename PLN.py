import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import numpy as np

# --- ส่วนที่ 1: การจำลองข้อมูล (Input Data) ---

# ข้อมูลสินค้าและกระบวนการผลิต (Routing & Standard Time)
# (Product_ID, Required_Machine_Type, Time_per_unit)
PRODUCTS = {
    "P001": {"machine_type": "CNC", "time_per_unit": 0.5, "material_cost_per_unit": 50},
    "P002": {"machine_type": "CNC", "time_per_unit": 0.7, "material_cost_per_unit": 65},
    "P003": {"machine_type": "Lathe", "time_per_unit": 1.0, "material_cost_per_unit": 80},
    "P004": {"machine_type": "Packing", "time_per_unit": 0.2, "material_cost_per_unit": 15},
}

# ข้อมูลเครื่องจักร (Machines)
# (Machine_ID, Machine_Type, Operating_Cost_per_hr, Setup_Time_hr)
MACHINES = {
    "M01": {"type": "CNC", "cost_per_hr": 300, "setup_time": 0.5},
    "M02": {"type": "CNC", "cost_per_hr": 320, "setup_time": 0.5},
    "M03": {"type": "Lathe", "cost_per_hr": 250, "setup_time": 1.0},
    "M04": {"type": "Packing", "cost_per_hr": 100, "setup_time": 0.1},
}

# ข้อมูลคำสั่งผลิต (Production Orders)
# (Order_ID, Product_ID, Quantity, Priority)
ORDERS = {
    "ORD01": {"product_id": "P001", "quantity": 10},
    "ORD02": {"product_id": "P003", "quantity": 5},
    "ORD03": {"product_id": "P002", "quantity": 8},
    "ORD04": {"product_id": "P004", "quantity": 20},
    "ORD05": {"product_id": "P001", "quantity": 15},
    "ORD06": {"product_id": "P003", "quantity": 7},
}

# --- ส่วนที่ 2: ระบบจำลองและฟังก์ชันประเมิน (Simulator & Fitness Function) ---

def simulate_schedule(plan):
    """
    จำลองการผลิตตามแผน (plan) ที่กำหนด และคำนวณตัวชี้วัด
    plan คือ list ของ (Order_ID, Machine_ID)
    """
    machine_availability = {m_id: 0 for m_id in MACHINES.keys()}
    machine_last_product = {m_id: None for m_id in MACHINES.keys()}
    
    schedule_details = []
    total_cost = 0
    total_labor_hours = 0

    for order_id, machine_id in plan:
        order = ORDERS[order_id]
        product = PRODUCTS[order["product_id"]]
        machine = MACHINES[machine_id]

        # ตรวจสอบว่าเครื่องจักรถูกต้องหรือไม่
        if product["machine_type"] != machine["type"]:
             # แผนนี้เป็นไปไม่ได้ ให้คะแนนต่ำสุด
            return None, float('inf'), float('inf'), float('inf')

        # คำนวณเวลาที่ต้องใช้
        processing_time = product["time_per_unit"] * order["quantity"]
        
        # คำนวณ Setup Time ถ้ามีการเปลี่ยน Product
        setup_time = 0
        if machine_last_product[machine_id] != order["product_id"]:
            setup_time = machine["setup_time"]
            machine_last_product[machine_id] = order["product_id"]

        start_time = machine_availability[machine_id]
        end_time = start_time + setup_time + processing_time
        
        machine_availability[machine_id] = end_time

        # คำนวณต้นทุน
        op_cost = (setup_time + processing_time) * machine["cost_per_hr"]
        mat_cost = product["material_cost_per_unit"] * order["quantity"]
        total_cost += (op_cost + mat_cost)
        
        # สมมติว่า 1 งานใช้คน 1 คนตลอดเวลา
        total_labor_hours += (setup_time + processing_time)

        schedule_details.append({
            "Order_ID": order_id,
            "Product_ID": order["product_id"],
            "Quantity": order["quantity"],
            "Machine_ID": machine_id,
            "Start_Time": start_time,
            "End_Time": end_time,
            "Setup_Time": setup_time,
            "Processing_Time": processing_time
        })
    
    # เวลาที่เสร็จสิ้นทั้งหมด (Makespan) คือเวลาที่เครื่องจักรตัวสุดท้ายทำงานเสร็จ
    makespan = max(machine_availability.values())

    return schedule_details, makespan, total_cost, total_labor_hours

def calculate_fitness(metrics, weights):
    """คำนวณ Fit Score จากตัวชี้วัดและน้ำหนักที่กำหนด"""
    makespan, total_cost, total_labor_hours = metrics

    # ทำให้เป็น Normalised Score (ค่ายิ่งน้อยยิ่งดี -> คะแนนยิ่งสูง)
    time_score = 1.0 / (1.0 + makespan)
    cost_score = 1.0 / (1.0 + total_cost)
    labor_score = 1.0 / (1.0 + total_labor_hours)
    
    # รวมคะแนนตามน้ำหนัก
    fitness = (weights['time'] * time_score +
               weights['cost'] * cost_score +
               weights['labor'] * labor_score)
    
    return fitness

# --- ส่วนที่ 3: แกนหลัก AI (Genetic Algorithm) ---

def create_random_plan():
    """สร้างแผนการผลิตแบบสุ่ม"""
    # 1. สุ่มลำดับของ Orders
    shuffled_orders = list(ORDERS.keys())
    random.shuffle(shuffled_orders)

    # 2. จับคู่แต่ละ Order กับ Machine ที่ถูกต้อง
    plan = []
    for order_id in shuffled_orders:
        product_id = ORDERS[order_id]["product_id"]
        required_machine_type = PRODUCTS[product_id]["machine_type"]
        
        # หาเครื่องจักรที่สามารถทำงานนี้ได้
        available_machines = [m_id for m_id, m_info in MACHINES.items() if m_info["type"] == required_machine_type]
        
        # สุ่มเลือก 1 เครื่องจากเครื่องที่ว่าง
        chosen_machine = random.choice(available_machines)
        plan.append((order_id, chosen_machine))
    return plan

def crossover(parent1, parent2):
    """สร้างลูกจากการผสมข้ามของพ่อแม่ (Single Point Crossover)"""
    point = random.randint(1, len(parent1) - 2)
    # ลูกจะได้รับส่วนหัวจากพ่อ1 และส่วนหางจากพ่อ2 แต่ต้องแน่ใจว่าไม่มี order ซ้ำ
    child1_part1 = parent1[:point]
    child1_order_ids = {o[0] for o in child1_part1}
    
    child1_part2 = [gene for gene in parent2 if gene[0] not in child1_order_ids]
    
    return child1_part1 + child1_part2

def mutate(plan, mutation_rate=0.1):
    """ปรับเปลี่ยนแผนเล็กน้อย (สลับตำแหน่งงาน)"""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(plan)), 2)
        plan[idx1], plan[idx2] = plan[idx2], plan[idx1]
    return plan

# --- ส่วนที่ 4: การรันโปรแกรมและแสดงผล ---

def plot_gantt_chart(schedule_df):
    """วาด Gantt Chart จากตารางแผนการผลิต"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # ระบุ path ฟอนต์ภาษาไทย (เช่น Tahoma หรือ Sarabun)
    # ตัวอย่างนี้ใช้ Tahoma ซึ่งมีใน Windows เกือบทุกเครื่อง
    font_path = r"C:\Windows\Fonts\tahoma.ttf"
    font_th = fm.FontProperties(fname=font_path)

    machines = sorted(schedule_df['Machine_ID'].unique())
    unique_orders = schedule_df['Order_ID'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_orders)))
    order_colors = {order: color for order, color in zip(unique_orders, colors)}

    for i, machine in enumerate(machines):
        machine_jobs = schedule_df[schedule_df['Machine_ID'] == machine]
        for _, job in machine_jobs.iterrows():
            start = job['Start_Time']
            duration = job['End_Time'] - start
            ax.barh(i, duration, left=start, height=0.6, align='center',
                    edgecolor='black', color=order_colors[job['Order_ID']],
                    label=job['Order_ID'])
            ax.text(start + duration/2, i, f"{job['Order_ID']}\n({job['Product_ID']})",
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8, fontproperties=font_th)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontproperties=font_th)
    ax.set_xlabel("เวลา (ชั่วโมง)", fontproperties=font_th)
    ax.set_ylabel("เครื่องจักร", fontproperties=font_th)
    ax.set_title("Gantt Chart แผนการผลิตที่ดีที่สุด", fontproperties=font_th)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("เลือกเป้าหมายหลักในการวางแผน:")
    print("1 = เวลาน้อยที่สุด, 2 = ต้นทุนต่ำที่สุด, 3 = ใช้แรงงานน้อยที่สุด")
    choice = input("กรุณาเลือก (1/2/3): ").strip()
    if choice == "1":
        OPTIMIZATION_WEIGHTS = {'time': 0.8, 'cost': 0.15, 'labor': 0.05}
    elif choice == "2":
        OPTIMIZATION_WEIGHTS = {'time': 0.1, 'cost': 0.8, 'labor': 0.1}
    elif choice == "3":
        OPTIMIZATION_WEIGHTS = {'time': 0.1, 'cost': 0.1, 'labor': 0.8}
    else:
        OPTIMIZATION_WEIGHTS = {'time': 0.6, 'cost': 0.3, 'labor': 0.1}
    print(f"เป้าหมายการทำงาน: เวลา {OPTIMIZATION_WEIGHTS['time']*100}%, ต้นทุน {OPTIMIZATION_WEIGHTS['cost']*100}%, แรงงาน {OPTIMIZATION_WEIGHTS['labor']*100}%\n")

    POPULATION_SIZE = 50
    NUM_GENERATIONS = 100
    ELITISM_SIZE = 5

    population = [create_random_plan() for _ in range(POPULATION_SIZE)]
    
    best_plan_so_far = None
    best_fitness_so_far = -1

    for gen in range(NUM_GENERATIONS):
        fitness_scores = []
        for plan in population:
            _, makespan, cost, labor = simulate_schedule(plan)
            if makespan == float('inf'):
                fitness = 0
            else:
                fitness = calculate_fitness((makespan, cost, labor), OPTIMIZATION_WEIGHTS)
            fitness_scores.append((fitness, plan))
        
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        if fitness_scores[0][0] > best_fitness_so_far:
            best_fitness_so_far = fitness_scores[0][0]
            best_plan_so_far = fitness_scores[0][1]

        if (gen + 1) % 10 == 0:
            print(f"Generation {gen+1}/{NUM_GENERATIONS} - Best Fitness: {best_fitness_so_far:.4f}")

        next_population = []
        elites = [p[1] for p in fitness_scores[:ELITISM_SIZE]]
        next_population.extend(elites)
        
        while len(next_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices([p[1] for p in fitness_scores[:POPULATION_SIZE//2]], k=2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_population.append(child)
        
        population = next_population

    print("\n--- การวิเคราะห์เสร็จสิ้น ---")
    print("แผนการผลิตที่ดีที่สุดที่พบ:")
    
    final_schedule, final_makespan, final_cost, final_labor = simulate_schedule(best_plan_so_far)
    
    schedule_df = pd.DataFrame(final_schedule)
    
    print("\n[ตารางแผนการผลิต]")
    print(schedule_df.to_string())
    
    print("\n[สรุปตัวชี้วัด]")
    print(f"  - เวลาที่ใช้ทั้งหมด (Makespan): {final_makespan:.2f} ชั่วโมง")
    print(f"  - ต้นทุนการผลิตทั้งหมด: {final_cost:,.2f} บาท")
    print(f"  - ชั่วโมงแรงงานทั้งหมด: {final_labor:.2f} ชั่วโมง")

    plot_gantt_chart(schedule_df)
