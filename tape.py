class Room:
    def __init__(self, name, initial_state=0):
        self.name = name
        self.state = initial_state  # 0 (out/inactive) or 1 (in/active)
    
    def toggle(self):
        """Toggle the room state between 0 and 1"""
        self.state = 1 - self.state
        return self.state
    
    def __repr__(self):
        return f"{self.name}: {'IN' if self.state == 1 else 'OUT'}"


class TapeDrive:
    def __init__(self):
        self.tape = []
    
    def record(self, room_states):
        """Record the current state of all rooms to the tape"""
        self.tape.append(room_states.copy())
        
    def get_history(self):
        """Get the full history of recorded states"""
        return self.tape


class LogicGateSystem:
    def __init__(self, rooms):
        self.rooms = rooms
        self.tape_drive = TapeDrive()
        self.record_current_state()
    
    def record_current_state(self):
        """Record the current state of all rooms to the tape drive"""
        state_dict = {room.name: room.state for room in self.rooms}
        self.tape_drive.record(state_dict)
    
    def get_active_rooms(self):
        """Return a set of rooms that are currently active (state=1)"""
        return {room.name for room in self.rooms if room.state == 1}
    
    def get_inactive_rooms(self):
        """Return a set of rooms that are currently inactive (state=0)"""
        return {room.name for room in self.rooms if room.state == 0}
    
    def get_room_state(self, room_name):
        """Get the state of a specific room by name"""
        for room in self.rooms:
            if room.name == room_name:
                return room.state
        raise ValueError(f"Room '{room_name}' not found")
    
    # Logic gate operations
    def AND_operation(self, room1_name, room2_name):
        """Perform AND operation on two rooms' states, return the result"""
        room1_state = self.get_room_state(room1_name)
        room2_state = self.get_room_state(room2_name)
        return room1_state and room2_state
    
    def OR_operation(self, room1_name, room2_name):
        """Perform OR operation on two rooms' states, return the result"""
        room1_state = self.get_room_state(room1_name)
        room2_state = self.get_room_state(room2_name)
        return room1_state or room2_state
    
    def NOT_operation(self, room_name):
        """Perform NOT operation on a room's state, return the result"""
        room_state = self.get_room_state(room_name)
        return 1 - room_state
    
    def XOR_operation(self, room1_name, room2_name):
        """Perform XOR operation on two rooms' states, return the result"""
        room1_state = self.get_room_state(room1_name)
        room2_state = self.get_room_state(room2_name)
        return (room1_state or room2_state) and not (room1_state and room2_state)
    
    # Set theory operations
    def union(self, set1, set2):
        """Return the union of two sets"""
        return set1.union(set2)
    
    def intersection(self, set1, set2):
        """Return the intersection of two sets"""
        return set1.intersection(set2)
    
    def complement(self, set1):
        """Return the complement of a set with respect to all rooms"""
        all_rooms = {room.name for room in self.rooms}
        return all_rooms - set1
    
    def execute_rule(self, rule):
        """Execute a rule that determines which room to toggle based on current states"""
        room_to_toggle = rule(self)
        if room_to_toggle:
            room = next((room for room in self.rooms if room.name == room_to_toggle), None)
            if room:
                room.toggle()
                self.record_current_state()
                return True
        return False
    
    def solve_problem(self, rule_set, max_steps=100):
        """Attempt to solve the problem using a set of rules"""
        step = 0
        while step < max_steps:
            step += 1
            made_change = False
            
            for rule in rule_set:
                if self.execute_rule(rule):
                    made_change = True
                    break
            
            if not made_change:
                print(f"Stabilized after {step} steps - no more applicable rules")
                break
        
        if step == max_steps:
            print(f"Reached maximum steps ({max_steps}) without stabilizing")
        
        return self.tape_drive.get_history()


# Example usage
def main():
    # Create rooms
    rooms = [
        Room("A"),  # Problem component A
        Room("B"),  # Problem component B
        Room("C"),  # Problem component C
        Room("D"),  # Problem component D
        Room("E")   # Problem component E
    ]
    
    # Initialize the logic gate system
    system = LogicGateSystem(rooms)
    
    # Define rules using logic gates and set theory
    # Each rule returns the name of a room to toggle, or None if no action needed
    def rule1(sys):
        # If A and B are active, toggle C
        if sys.AND_operation("A", "B"):
            return "C"
        return None
    
    def rule2(sys):
        # If C is active and D is inactive, toggle E
        if sys.get_room_state("C") == 1 and sys.get_room_state("D") == 0:
            return "E"
        return None
    
    def rule3(sys):
        # If the active rooms set intersects with {A, E}, toggle D
        active_rooms = sys.get_active_rooms()
        if len(sys.intersection(active_rooms, {"A", "E"})) > 0:
            return "D"
        return None
    
    def rule4(sys):
        # If we're in an odd number of rooms, toggle A
        if len(sys.get_active_rooms()) % 2 == 1:
            return "A"
        return None
    
    def rule5(sys):
        # If we're in less than 3 rooms total, toggle B
        if len(sys.get_active_rooms()) < 3:
            return "B"
        return None
    
    # Set up the initial state (activate room A)
    rooms[0].toggle()  # Activate room A
    system.record_current_state()
    
    # Define the rule set
    rule_set = [rule1, rule2, rule3, rule4, rule5]
    
    # Solve the problem
    history = system.solve_problem(rule_set)
    
    # Print the solution steps
    print("Solution steps:")
    for i, state in enumerate(history):
        print(f"Step {i}: {state}")


if __name__ == "__main__":
    main()