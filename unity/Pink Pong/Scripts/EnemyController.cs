using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyController : MonoBehaviour
{
    public float moveSpeed; // The moving speed of the enemy
    private float randomYStartPos; // Both player and enemy start at a random Y position
    public float sensorOffset = 0.2f; // A small offset for better functionality of the three 'bat' sensors
    public float upperBorderPos = 2.8f; // Since the bats have no rigid bodies, we need to avoid 'crashes' with upper lower sides
    public float lowerBorderPos = -4.75f;
    public Vector3 upperBorderOffset = new Vector3(0, -0.2f, 0); // If a crash with upper or lower side happens, the bat is moved back a certain distance
    public Vector3 lowerBorderOffset = new Vector3(0, 0.2f, 0);
    public GameObject sensorUpper; // Initializing the three sensors that bats are equipped with
    public GameObject sensorMiddle;
    public GameObject sensorLower;
    public GameObject ball;
    public BallController ballController;
    public Transform target; // Initializing the ball as 'target' the enemy is moving towards
    // Start is called before the first frame update
    void Start()
    {
        ballController = ball.gameObject.GetComponent<BallController>();
        moveSpeed = ballController.enemyMoveSpeed;
        randomYStartPos = Random.Range(-4.5f, 2.5f);
        transform.position = new Vector3(transform.position.x, randomYStartPos, transform.position.z);
    }

    // Update is called once per frame
    void Update()
    {
        // Placing the sensors directly on the bats
        sensorUpper.transform.position = new Vector3(sensorUpper.transform.position.x, transform.position.y + sensorOffset, sensorUpper.transform.position.z);
        sensorMiddle.transform.position = new Vector3(sensorMiddle.transform.position.x, transform.position.y, sensorMiddle.transform.position.z);
        sensorLower.transform.position = new Vector3(sensorLower.transform.position.x, transform.position.y - sensorOffset, sensorLower.transform.position.z);

        // The speed of moving towards the ball
        float step = moveSpeed * Time.deltaTime;

        // The enemy is moving at a realistic speed 'with' the ball's position on the Y axis
        transform.position = Vector3.MoveTowards(transform.position, new Vector3(transform.position.x, target.position.y, transform.position.z), step);

        // Avoiding 'crashes' with the upper and lower side
        if (transform.position.y == upperBorderPos)
        {
            transform.position = transform.position + upperBorderOffset;
        }

        if (transform.position.y == lowerBorderPos)
        {
            transform.position = transform.position + lowerBorderOffset;
        }
    }
}

//need to work on sensors and border movement control

