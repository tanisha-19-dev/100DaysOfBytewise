class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def is_palindrome(head):
    def find_middle(head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def reverse(head):
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev

    if not head or not head.next:
        return True

    middle = find_middle(head)

    second_half_start = reverse(middle)

    first_half_start = head
    second_half_iter = second_half_start
    while second_half_iter:
        if first_half_start.val != second_half_iter.val:
            return False
        first_half_start = first_half_start.next
        second_half_iter = second_half_iter.next

    reverse(second_half_start)

    return True

# Helper function to create a linked list from a list of values
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head

def print_palindrome_result(values):
    head = create_linked_list(values)
    if is_palindrome(head):
        print("The linked list is a palindrome.")
    else:
        print("The linked list is not a palindrome.")

print_palindrome_result([1, 2, 3, 2, 1])  # The linked list is a palindrome.
print_palindrome_result([1, 2, 3, 4, 5])  # The linked list is not a palindrome.
