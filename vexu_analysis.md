### The APIs related to VEXU Android APP
- **VEXU_API_BASE_URL_DEV**="http://app.vexu.ai/api/v1/ai/calls"
- **VEXU_API_TOKEN_DEV**="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhODNmM2U3My1jM2U1LTQ4ZjktYTBlNy1kOTRlMzU3MjRlMmIiLCJpc19haV9zZXJ2aWNlIjp0cnVlfQ.Lp1r8WvvvijNo8ilGCs5bbqwql9BQuq929yUCp-b40g"
- **VEXU_USERS_API_BASE_URL_DEV**="https://app.vexu.ai/api/v1/ai/users/by-phone/"
- **VEXU_CALLER_NAME_CONSTANT_DEV**="Unknown Caller (Dev)"

### Functions related to VEXU Android APP
- **get_vexu_user_details(phone_number: str)**: Fetches user details (name and user_id) from Vexu Users API.
- **post_to_vexu_api(endpoint: str, vexu_call_id: str = None, data: dict = None)**: This function sends data to the app, the endpoint arg is the user identifier.
- **post_vexu_start_call(twilio_call_sid: str, caller_phone: str, dynamic_vexu_user_id: str)**: This function creates an initial payload and creates the vexu_call_id(from uuid.uuid4) and sets the time then it will be sent posted via `post_to_vexu_api` API. It is called at the beginning of `handle_media_stream` i.e at the begining of each call.
    ```json
        payload = {
            "user_id": dynamic_vexu_user_id,
            "call_sid": vexu_call_id_generated,
            "contact_id": None,
            "caller_name": None,
            "caller_phone": caller_phone if caller_phone else "Unknown",
            "start_time": start_time_iso,
            "end_time": start_time_iso,
            "duration": 1,
            "transcript": "",
            "audio_base64": "",
            "summary": "",
            "is_encrypted": False,
            "is_emergency": False,  # Initialize as False
        }
    ```
- **post_vexu_message_async(vexu_call_id: str, text: str, sender: str = "caller", audio_pcm16_8khz_bytes: bytes = None,)**: It sends the sender(agent or the caller) voice with its transcription to the app(via `post_to_vexu_api` API)
    ```json
        payload = {
        "sender": sender,
        "text": text,
        "timestamp": timestamp_iso,
        "message_type": "regular",
        "audio_base64": audio_base64_payload,
        "is_encrypted": False,
        }
    ```
    for example:
    ```json
        post_vexu_message_async(
        vexu_call_id=current_vexu_call_id,
        text=response["transcript"],
        sender="agent",
        audio_pcm16_8khz_bytes=agent_audio_for_vexu_pcm16_8khz,
        )
    ```

- **post_vexu_end_call(vexu_call_id: str)**: This function posts to /vexu_call_id/end in the APP
- **update_vexu_call_emergency_status(vexu_call_id: str, is_emergency_status: bool)**: This function updates the information and updates the emergency field.
- **get_vexu_call_details(vexu_call_id: str)**: get all the information
- **update_vexu_call_summary(vexu_call_id: str, summary_text: str, caller_name_override: Optional[str] = None, existing_call_data_fetched: Optional[Dict[str, Any]] = None,)** Updates the summary
- **get_vexu_messages(vexu_call_id: str)**: Gets all of the messages from app vexu
- **process_call_summary_and_update( vexu_call_id: str, twilio_call_sid: Optional[str], caller_name_for_summary: Optional[str],):**: gets all messages and then summarize.