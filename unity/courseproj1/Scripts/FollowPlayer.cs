using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowPlayer : MonoBehaviour
{
    public GameObject player;
    // Public variables are accessible even outside the class, private variables just inside the encompassing class
    // Public variables can be modified in the Unity Editor, private ones can't    
    [SerializeField] private Vector3 offset = new Vector3(0, 5, -7);
    // Start is called before the first frame update

    // No Start function here

    // Update is called once per frame

    // FixedUpdate in PlayerController, together with LateUpdate in FollowPlayer guarantees a fluid, non-choppy camera movement
    // Here, what's in LateUpdate is updated after FixedUpdate, so first the physics then the camera positioning
    void LateUpdate()
    {
        // Offset camera position to follow behind the vehicle by adding an offset vector (mirroring the camera's initial position)
        transform.position = player.transform.position + offset;
    }
}
