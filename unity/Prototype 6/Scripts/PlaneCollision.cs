using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlaneCollision : MonoBehaviour
{
    private Rigidbody ballRb;
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnCollisionEnter(Collision other)
    {
        // Once a spawned ball hits the ground (and not the crate) it is sent off 'into' the scene
        // This creates an interesting visual effect and avoids dozens or hundreds of ball lying around
        if (other.gameObject.CompareTag("Ball"))
        {
            ballRb = other.gameObject.GetComponent<Rigidbody>();
            ballRb.AddForce(Vector3.left * 5, ForceMode.Impulse);
        }
    }
}
