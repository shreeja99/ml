# Monthly Sales Visualization using Matplotlib
import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May"]
attendance = [85, 90, 78, 92, 88]

plt.plot(months, attendance, marker="o")
plt.title("Monthly Student Attendance")
plt.xlabel("Month")
plt.ylabel("Attendance (%)")
plt.show()
