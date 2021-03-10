using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float speed = 10.0f;
    private Rigidbody playerRb;
    private GameObject focalPoint;
    private float powerupStrength = 15.0f;
    public bool hasPowerup;
    public GameObject powerupIndicator;
    // Start is called before the first frame update
    void Start()
    {
        playerRb = GetComponent<Rigidbody>();
        focalPoint = GameObject.Find("Focal Point");
    }

    // Update is called once per frame
    void Update()
    {
        float forwardInput = Input.GetAxis("Vertical");
        // The player moves with the direction of the focal point
        playerRb.AddForce(focalPoint.transform.forward * speed * forwardInput);
        // The Powerup Indicator is placed directly 'on' the player once powerup has been received
        powerupIndicator.transform.position = transform.position + new Vector3(0, -0.5f, 0);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Powerup"))
        {
            hasPowerup = true;
            Destroy(other.gameObject);
            // We switch on the Powerup Indicator once the powerup has been received
            powerupIndicator.gameObject.SetActive(true);
            // Once the player has received the powerup, we start the countdown (the coroutine)
            StartCoroutine(PowerupCountdownRoutine());
        }
    }

    // Initialization of a Coroutine, a coding mechanism that allows some code to be executed outside of the Update method or Update loop
    // Here, we set up a timer which validates the received powerup for a given period of time.
    // Once that period of time has passed, the powerup gets deactivated.
    IEnumerator PowerupCountdownRoutine()
    {
        yield return new WaitForSeconds(7);
        hasPowerup = false;
        // We switch off the Powerup Indicator once the powerup has been lost due to timeout
        powerupIndicator.gameObject.SetActive(false);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Enemy") && hasPowerup)
        {
            Rigidbody enemyBody = collision.gameObject.GetComponent<Rigidbody>();
            // We subtract the player's position from the enemy's position so that the enemy can 'fly away' after the player has been equipped with a Powerup.
            Vector3 awayFromPlayer = (collision.gameObject.transform.position - transform.position);
            Debug.Log("Collided with: " + collision.gameObject.name + " with powerup set to " + hasPowerup);
            enemyBody.AddForce(awayFromPlayer * powerupStrength, ForceMode.Impulse);
        }
    }
}
