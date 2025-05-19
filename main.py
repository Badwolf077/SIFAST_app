import pypulse


if __name__ == "__main__":
    fiber_array = pypulse.FiberArray(0, 0)
    print(fiber_array.get_fiber_array_properties())
