using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class PlayerController : MonoBehaviour
{
    [SerializeField] private float horsePower = 0f;
    [SerializeField] private float turnSpeed = 45.0f;
    [SerializeField] float speed;
    [SerializeField] float rpm;
    [SerializeField] List<WheelCollider> allWheels;
    [SerializeField] int wheelsOnGround;
    public TextMeshProUGUI speedometerText;
    public TextMeshProUGUI rpmText;
    // Optional additional 'jumping' functionality (further implementation see below)
    private float jumpSpeed = 45.0f;
    private float horizontalInput;
    private float forwardInput;
    private float jumpInput;
    private Rigidbody playerRb;
    [SerializeField] GameObject centerOfMass;
    // Optionally add acceleration
    // private float acc = 1.0f;
    // Start is called before the first frame update
    
    void Start()
    {
        playerRb = GetComponent<Rigidbody>();
        playerRb.centerOfMass = centerOfMass.transform.position; // Initialize a Center of Mass variable to stabilize the car's pose
    }

    // Update is called once per frame

    // FixedUpdate in PlayerController, together with LateUpdate in FollowPlayer guarantees a fluid, non-choppy camera movement
    // Here, what's in LateUpdate is updated after FixedUpdate, so first the physics then the camera positioning
    void FixedUpdate()
    {
        // Horizontal and vertical inputs can have a max value of 1 (i.e. if respective buttons are being held down)
        horizontalInput = Input.GetAxis("Horizontal");
        forwardInput = Input.GetAxis("Vertical");
        jumpInput = Input.GetAxis("Jump");
        // Increase acceleration
        // acc += 0.001f;
        // Move the vehicle forward
        //transform.Translate(Vector3.forward * Time.deltaTime * speed * forwardInput);
        if (IsOnGround())
        {
            playerRb.AddRelativeForce(Vector3.forward * horsePower * forwardInput);
            // equally possible: define the x, y, z coordinates manually (Vector3.forward is a shortcut for 0, 0, 1, meaning 0 right-left, 0 up-down,
            // 1 forward-backward). To make movement slower, multiply by the actual time change between the frame updates (e.g. one frame changes in 0.1
            // s). The coordinates therefore mean the x, y, z change in meters per frame. If a frame updates in 0.1 s and z is set to 1, then in one
            // second, we see 10 meters forward movement.
            // transform.Translate(0, 0, 1 * Time.deltaTime * 20);

            // At the end, we multiply by the actual keyboard input. If no input perceived, there's no turning (multiply by zero)

            // Rotate left-right movement instead of unrealistic translation (a car cannot simply move sideways)
            transform.Rotate(Vector3.up, Time.deltaTime * turnSpeed * horizontalInput);
            transform.Translate(Vector3.up * Time.deltaTime * jumpSpeed * jumpInput);

            speed = Mathf.RoundToInt(playerRb.velocity.magnitude * 3.6f); // We need to multiply by 3.6 to get km/h
            speedometerText.text = "Speed: " + speed + " km/h"; // Display the speed in Game View

            rpm = (speed % 30) * 40;
            rpmText.text = "RPM: " + rpm;
        }
    }

    // Check if all wheels are grounded, if not (see conditional wrapper above) do not accelerate and thus do not display km/h and rpm
    bool IsOnGround()
    {
        wheelsOnGround = 0;
        
        foreach (WheelCollider wheel in allWheels)
        {
            if (wheel.isGrounded)
            {
                wheelsOnGround++;
            }
        }

        if (wheelsOnGround == 4)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}
